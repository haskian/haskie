"""Does Claude reach for haskie when a prompt never mentions it? Activation only: no task to
finish, no grader, a few turns per run.

Each prompt in `prompts.jsonl` is a plain user request, marked whether it should reach for the
library: questions on topics the book collection covers (some Claude believes it knows, the hard
case), and controls haskie's own rule says to leave alone - arithmetic, trivia, a poem, code in
front of it. Each runs under each setup:

    none       haskie connected over MCP, nothing else (arm G of `evals/run.py`)
    rule       plus the rule and skill `haskie install claude` writes (arm I)
    hook       plus the `UserPromptSubmit` hook that injects haskie's top passages (arm H)
    rule+hook  both

A run stops after `MAX_TURNS`: loading a deferred tool and calling it fit, an answer does not
need to finish. It records whether Claude called a haskie search (`searched`), and for the hook
setups whether the hook had passages to inject (`nudged`) - worked out by making the hook's own
search, since hook output does not reach the transcript. The report gives, per setup, how often
the library was consulted on the prompts that should reach for it, how often it was on the
controls (a false activation), and the cost. Runs are isolated like the agent eval's: no auto
memory, a working directory outside the home, denied reads, searches scoped to one collection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import msgspec

from evals import run, scope, steer

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts.jsonl"
REPORTS = HERE / "reports"
MAX_TURNS = 4
SETUPS = {  # name -> (rule installed, hook on)
    "none": (False, False),
    "rule": (True, False),
    "hook": (False, True),
    "rule+hook": (True, True),
}
SEARCH = "mcp__haskie__search_"
LIMIT_MARKER = "session limit"


class Prompt(msgspec.Struct, forbid_unknown_fields=True):
    id: str
    prompt: str
    activate: bool  # whether it should reach for the library
    why: str


class Trial(msgspec.Struct):
    id: str
    setup: str
    sample: int
    activate: bool
    tools: list[str]  # every tool called, in order
    searched: bool  # Claude called a haskie search
    nudged: bool  # the hook had passages to inject (hook setups only)
    turns: int
    usd: float
    seconds: float
    limited: bool  # no result: the session limit stopped it, or it never ran (not a result)

    @property
    def consulted(self) -> bool:
        return self.searched or self.nudged


def load(path: Path = PROMPTS) -> list[Prompt]:
    decoder = msgspec.json.Decoder(Prompt)
    return [decoder.decode(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read(transcript: str) -> tuple[list[str], int, float, bool]:
    """From a stream-json transcript: the tools called in order, the turns, the cost, and whether
    it ended without a result - stopped by the session limit, or never run at all (an empty
    transcript), which must not count as "did not search"."""
    tools: list[str] = []
    turns, usd, limited = 0, 0.0, True
    for line in transcript.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = event.get("message")
        if event.get("type") == "assistant" and isinstance(message, dict):
            tools += [c["name"] for c in message.get("content", []) if c.get("type") == "tool_use"]
        if event.get("type") == "result":
            turns, usd = event.get("num_turns") or 0, event.get("total_cost_usd") or 0.0
            limited = LIMIT_MARKER in str(event.get("result", ""))  # a result: ran, unless limited
    return tools, turns, usd, limited


def would_nudge(prompt: str, api: str, collection: str) -> bool:
    """Whether the hook has passages to inject for `prompt`: its own questions and search."""
    asked = steer.questions(prompt)
    try:
        return bool(asked and steer.search(api, collection, asked))
    except Exception:  # the hook prints nothing on a failure, so it nudges nothing
        return False


def trial(
    prompt: Prompt, setup: str, sample: int, model: str, api: str, collection: str, out: Path
) -> Trial:
    rule, hook = SETUPS[setup]
    out = out.resolve()  # Claude runs elsewhere: a relative settings path would not be found
    directory = out / setup / prompt.id / str(sample)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    work = run.WORK_ROOT / "activation" / directory.relative_to(out)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    if rule:
        run.install_haskie(work, api, collection)
    settings = directory / "settings.json"
    settings.write_text(json.dumps(run.agent_settings("h" if hook else "g", api, collection)))
    started = time.monotonic()
    with scope.Proxy(api, collection) as proxy:
        mcp = directory / "mcp.json"
        mcp.write_text(json.dumps({"mcpServers": {"haskie": {"type": "http", "url": proxy.url}}}))
        argv = [run.claude_binary(), "-p", prompt.prompt, "--output-format", "stream-json"]
        argv += ["--verbose", "--setting-sources", "project", "--strict-mcp-config"]
        argv += ["--mcp-config", str(mcp), "--settings", str(settings)]
        argv += ["--allowedTools", *run.allowed_tools("g")]
        argv += ["--model", model, "--max-turns", str(MAX_TURNS)]
        with (directory / "transcript.jsonl").open("wb") as transcript:
            subprocess.run(
                argv,
                cwd=work,
                stdout=transcript,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                env=run.subprocess_environment(),
                check=False,
            )
    seconds = time.monotonic() - started
    shutil.rmtree(work, ignore_errors=True)
    tools, turns, usd, limited = read((directory / "transcript.jsonl").read_text(errors="replace"))
    return Trial(
        id=prompt.id,
        setup=setup,
        sample=sample,
        activate=prompt.activate,
        tools=tools,
        searched=any(name.startswith(SEARCH) for name in tools),
        nudged=hook and would_nudge(prompt.prompt, api, collection),
        turns=turns,
        usd=usd,
        seconds=seconds,
        limited=limited,
    )


def render(trials: list[Trial], setups: list[str]) -> str:
    counted = [t for t in trials if not t.limited]
    lines = [
        "# haskie activation",
        "",
        "`consulted`: Claude searched haskie, or the hook injected its passages. `searched`: "
        "Claude itself called a haskie search. Should-trigger prompts are on topics the "
        "collection covers; controls are ones haskie's rule says to leave alone.",
        "",
        "| setup | should trigger: consulted | searched | controls: consulted | searched | $/run |",
        "|---|---|---|---|---|---|",
    ]
    for setup in setups:
        mine = [t for t in counted if t.setup == setup]
        should = [t for t in mine if t.activate]
        controls = [t for t in mine if not t.activate]

        def rate(group: list[Trial], attribute: str) -> str:
            return f"{sum(getattr(t, attribute) for t in group)}/{len(group)}"

        usd = sum(t.usd for t in mine) / len(mine) if mine else 0.0
        lines.append(
            f"| {setup} | {rate(should, 'consulted')} | {rate(should, 'searched')} "
            f"| {rate(controls, 'consulted')} | {rate(controls, 'searched')} | {usd:.3f} |"
        )
    lines += ["", "## Per prompt: searched / runs", ""]
    lines.append("| prompt | should | " + " | ".join(setups) + " |")
    lines.append("|---|---|" + "---|" * len(setups))
    for pid in dict.fromkeys(t.id for t in counted):
        row = [t for t in counted if t.id == pid]
        cells = []
        for setup in setups:
            group = [t for t in row if t.setup == setup]
            cells.append(f"{sum(t.searched for t in group)}/{len(group)}" if group else "-")
        lines.append(
            f"| {pid} | {'yes' if row[0].activate else 'no'} | " + " | ".join(cells) + " |"
        )
    limited = len(trials) - len(counted)
    if limited:
        lines += [
            "",
            f"{limited} runs ended without a result (session limit, or never started): left out.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setups", nargs="+", choices=list(SETUPS), default=list(SETUPS))
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--model", default=os.environ.get("EVAL_MODEL", "sonnet"))
    parser.add_argument("--api", default=os.environ.get("HASKIE_EVAL_URL", "http://127.0.0.1:8123"))
    parser.add_argument("--collection", default="eval-programming-books")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--only", nargs="+", metavar="ID", help="run these prompt ids only")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--out", type=Path, default=REPORTS / stamp)
    args = parser.parse_args(argv)
    prompts = [p for p in load(args.prompts) if not args.only or p.id in args.only]
    trials: list[Trial] = []
    for setup in args.setups:
        for prompt in prompts:
            for sample in range(args.samples):
                done = trial(prompt, setup, sample, args.model, args.api, args.collection, args.out)
                trials.append(done)
                mark = "limited" if done.limited else ("searched" if done.searched else "-")
                print(f"{setup:10} {prompt.id:22} {sample}  {mark}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "trials.jsonl").write_text(
        "".join(msgspec.json.encode(t).decode() + "\n" for t in trials), encoding="utf-8"
    )
    text = render(trials, args.setups)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {args.out}")
    return 1 if any(t.limited for t in trials) else 0


if __name__ == "__main__":
    sys.exit(main())
