"""A seeded synthetic corpus: one fictional organization's service runbooks, and the tasks that ask
about them.

The book tasks all share a flaw for measuring haskie: the model already knows the books. These
facts are drawn at random per seed, so nothing about them can be recalled or guessed, and the
corpus can be regenerated at will. Each task asks for four documented facts about one target
service, and the task set is a grid over two dials:

- size: the target runbook plus distractors that look just like it - 5, 50, 500 or 2000
  documents in all, nested, so the only thing that changes between sizes is how much there is to
  sift;
- level: how the task refers to the target. 0 names it; 1 describes it in the runbook's own words;
  2 describes it in different words - no content word of the description shares a stem with the
  target's purpose line, which `_assert_paraphrases_disjoint` checks over every possible target.

Level 2 is where keyword matching - grep, or haskie without an embedding profile - should fall
behind semantic search. File names are neutral (`doc-s1-0137.md`), since a descriptive name let
arm D solve the book tasks from a directory listing alone.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import shutil
import textwrap
import zlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYNTH_DIR = ROOT / "synth"
TASK_ROOT = ROOT / "tasks"

SIZES = (5, 50, 500, 2000)
# `services` draws this many; sizes past it extend the corpus (`extra_services`) without
# disturbing the first 500, which earlier results were measured on.
BASE_POOL = 500
LEVELS = (0, 1, 2)
DEFAULT_SEED = 1
# Every seed setup builds and the harness can run. A second seed replicates the findings on a
# different organization: other targets, other decoys, other values, same dials.
SEEDS = (1, 2)

# (runbook wording, paraphrase), with {obj} and {cp} slots in the actions
ACTIONS = (
    ("reconciles {obj} against {cp}", "makes sure {obj} agree with {cp}"),
    ("exports {obj} to {cp}", "sends {obj} along to {cp}"),
    ("imports {obj} from {cp}", "pulls in {obj} coming from {cp}"),
    ("archives {obj} received from {cp}", "keeps long-term copies of {obj} sent by {cp}"),
    ("validates {obj} submitted by {cp}", "checks for mistakes in {obj} handed over by {cp}"),
    (
        "deduplicates {obj} arriving from {cp}",
        "removes repeated entries in {obj} delivered by {cp}",
    ),
)
OBJECTS = (
    ("the general ledger", "our accounting books"),
    ("invoice records", "the bills we issue to clients"),
    ("payroll batches", "salary payment runs"),
    ("shipment manifests", "lists of what each parcel contains"),
    ("customer addresses", "where our clients live"),
    ("refund requests", "money-back claims"),
    ("tax filings", "what we report to the revenue office"),
    ("inventory counts", "how much stock sits on each depot's shelves"),
)
COUNTERPARTIES = (
    ("the card processor", "the firm that handles plastic-money transactions"),
    ("the bank feed", "the statement stream from our lender"),
    ("the warehouse system", "the software that runs our storage depots"),
    ("the tax authority portal", "the government's revenue website"),
    ("the carrier API", "the parcel courier's integration"),
    ("the HR platform", "the people team's tool"),
)
SCHEDULES = (
    ("every night", "once a day after the office closes"),
    ("every hour", "at sixty-minute intervals"),
    ("every fifteen minutes", "four times per hour"),
    ("on the first day of each month", "at the start of every billing cycle"),
    ("whenever a new file lands", "as soon as fresh data shows up"),
    ("every Monday morning", "at the start of each working week"),
)

BIRDS = (
    "heron", "kestrel", "osprey", "plover", "wren", "finch", "egret", "ibis", "lark", "martin",
    "swift", "tern", "crane", "raven", "robin", "gannet", "curlew", "dunlin", "merlin", "shrike",
    "avocet", "bittern", "grebe", "sanderling",
)  # fmt: skip
MINERALS = (
    "basalt", "quartz", "gneiss", "schist", "slate", "flint", "marble", "jasper", "onyx", "opal",
    "garnet", "beryl", "agate", "mica", "talc", "pumice", "shale", "chert", "galena", "pyrite",
    "zircon", "spinel", "topaz", "cobalt", "nickel",
)  # fmt: skip
# Names for services past `BASE_POOL` (fish x stars) - not trees, which the team names use.
FISH = (
    "perch", "pike", "trout", "salmon", "carp", "bream", "tench", "roach", "dace", "chub",
    "gudgeon", "minnow", "sturgeon", "halibut", "mackerel", "herring", "sardine", "anchovy",
    "turbot", "plaice", "sole", "flounder", "haddock", "pollock", "hake", "whiting", "grayling",
    "char", "smelt", "shad", "bass", "barbel", "loach", "ruffe", "goby", "wrasse", "mullet",
    "gurnard", "snapper", "tarpon",
)  # fmt: skip
STARS = (
    "vega", "altair", "deneb", "rigel", "sirius", "castor", "pollux", "spica", "antares",
    "arcturus", "capella", "procyon", "betelgeuse", "aldebaran", "regulus", "mira", "algol",
    "mizar", "alcor", "polaris", "canopus", "achernar", "hadar", "acrux", "mimosa", "shaula",
    "bellatrix", "alnilam", "alnitak", "mintaka", "saiph", "electra", "maia", "merope",
    "taygeta", "atlas", "alcyone", "celaeno", "sterope", "nunki",
)  # fmt: skip
TEAMS = (
    "Team Alder", "Team Birch", "Team Cedar", "Team Hazel", "Team Juniper", "Team Larch",
    "Team Maple", "Team Rowan", "Team Spruce", "Team Willow", "Team Yew", "Team Aspen",
)  # fmt: skip
TIMEOUTS = (5, 10, 15, 20, 30, 45, 60, 90, 120)

# Shared by every runbook, so matching it tells a search nothing. None of it may use a purpose
# term (checked in `_assert_boilerplate_neutral`), or grepping one would match every document.
BOILERPLATE = (
    "Deploys go out through the standard pipeline. A deploy that fails its health checks is "
    "rolled back automatically; the rollback is announced in the operations channel.",
    "Logs are shipped to the central log store and kept for thirty days. Include the request id "
    "from the response headers when you ask for help with a specific call.",
    "Alerts fire when the error rate stays above two percent for a sustained stretch, or when "
    "the queue depth doubles in a short window. Both page the owning team.",
    "Configuration changes need a second reviewer. Values in the table below are the ones in "
    "production; staging mirrors them unless a ticket says otherwise.",
    "Dependencies are pinned and upgraded on a regular cadence by the infrastructure group. "
    "Security patches skip that cadence and ship as soon as they are reviewed.",
    "Capacity is reviewed each quarter. If sustained load passes seventy percent of the "
    "provisioned limit, open a ticket with the capacity planning group.",
)

STOPWORDS = frozenset(
    "the and that with from into over each every what where when whenever once this their "
    "our your has have per".split()
)


@dataclass(frozen=True)
class Service:
    name: str
    purpose: tuple[int, int, int, int]  # indexes into ACTIONS, OBJECTS, COUNTERPARTIES, SCHEDULES
    port: int
    max_attempts: int
    timeout_seconds: int
    team: str


def describe(purpose: tuple[int, int, int, int], paraphrased: bool) -> str:
    side = 1 if paraphrased else 0
    action, obj, cp, schedule = purpose
    body = ACTIONS[action][side].format(obj=OBJECTS[obj][side], cp=COUNTERPARTIES[cp][side])
    return f"{body}, {SCHEDULES[schedule][side]}"


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 4 and w not in STOPWORDS}


def _share_stem(a: str, b: str) -> bool:
    return a[:4] == b[:4]


def _assert_paraphrases_disjoint() -> None:
    """For every possible target, no content word of its paraphrased description shares a
    four-letter stem with its runbook purpose line - otherwise level 2 is level 1 in disguise."""
    for purpose in itertools.product(
        range(len(ACTIONS)), range(len(OBJECTS)), range(len(COUNTERPARTIES)), range(len(SCHEDULES))
    ):
        doc, para = _words(describe(purpose, False)), _words(describe(purpose, True))
        clashes = {(p, d) for p in para for d in doc if _share_stem(p, d)}
        if clashes:
            raise AssertionError(f"paraphrase leaks runbook wording for {purpose}: {clashes}")
    names = set(BIRDS) | set(MINERALS) | set(FISH) | set(STARS)
    for phrase in (*ACTIONS, *OBJECTS, *COUNTERPARTIES, *SCHEDULES):
        clashes = {(p, n) for p in _words(phrase[1]) for n in names if _share_stem(p, n)}
        if clashes:
            raise AssertionError(f"paraphrase shares a stem with a service name: {clashes}")


def _assert_boilerplate_neutral() -> None:
    terms = set().union(
        *(_words(phrase[0]) for phrase in (*ACTIONS, *OBJECTS, *COUNTERPARTIES, *SCHEDULES))
    )
    for paragraph in BOILERPLATE:
        clashes = {t for t in terms if t in paragraph.lower()}
        if clashes:
            raise AssertionError(f"boilerplate repeats purpose terms {clashes}: {paragraph!r}")


def services(seed: int, count: int) -> list[Service]:
    """`count` services with distinct names, ports and purposes. The first is the target; the
    next four each differ from it in exactly one part of its purpose, so even the 5-document
    corpus has close decoys, and a fifth of the rest are near misses too."""
    rng = random.Random(seed)
    names = rng.sample([f"{b}-{m}" for b in BIRDS for m in MINERALS], count)
    ports = rng.sample(range(20000, 30000), count)
    dims = (len(ACTIONS), len(OBJECTS), len(COUNTERPARTIES), len(SCHEDULES))

    def draw() -> tuple[int, int, int, int]:
        action, obj, cp, schedule = (rng.randrange(d) for d in dims)
        return (action, obj, cp, schedule)

    target = draw()
    purposes = [target]
    used = {target}

    def near_miss(axis: int) -> tuple[int, int, int, int]:
        other = list(target)
        other[axis] = rng.choice([v for v in range(dims[axis]) if v != target[axis]])
        return (other[0], other[1], other[2], other[3])

    while len(purposes) < count:
        if len(purposes) < 5:
            candidate = near_miss(len(purposes) - 1)
        elif rng.random() < 0.2:
            candidate = near_miss(rng.randrange(4))
        else:
            candidate = draw()
        if candidate not in used:
            used.add(candidate)
            purposes.append(candidate)
    return [
        Service(
            names[i],
            purposes[i],
            ports[i],
            rng.randint(2, 9),
            rng.choice(TIMEOUTS),
            rng.choice(TEAMS),
        )
        for i in range(count)
    ]


def runbook(service: Service, rng: random.Random, pointer: str = "") -> str:
    """`pointer` is a line some kinds add - where to look beyond this page. Empty leaves the
    base runbook byte-for-byte as it was, which the already-imported documents depend on."""
    notes = "\n\n".join(rng.sample(BOILERPLATE, 2))
    purpose = describe(service.purpose, False)
    return (
        f"# {service.name}\n\n"
        f"## Purpose\n\n{purpose[0].upper()}{purpose[1:]}.\n\n"
        f"## Ownership\n\nOwned by {service.team}. Pages route to the owning team's on-call "
        "rotation; see the escalation policy for after-hours coverage.\n\n"
        "## Configuration\n\n"
        "| setting | value |\n|---|---|\n"
        f"| listen port | {service.port} |\n"
        f"| max retry attempts | {service.max_attempts} |\n"
        f"| request timeout | {service.timeout_seconds} seconds |\n\n"
        f"## Operational notes\n\n{notes}\n" + (f"\n{pointer}\n" if pointer else "")
    )


def doc_name(seed: int, index: int) -> str:
    return f"doc-s{seed}-{index:04d}.md"


def collection_name(seed: int, size: int, kind: str | None = None) -> str:
    return f"eval-synth-s{seed}-{kind}-n{size}" if kind else f"eval-synth-s{seed}-n{size}"


def corpus_dir(seed: int, size: int, kind: str | None = None) -> Path:
    base = SYNTH_DIR / f"seed-{seed}"
    return base / kind / f"size-{size}" if kind else base / f"size-{size}"


def task_name(seed: int, size: int, level: int) -> str:
    return f"synth_s{seed}_n{size}_l{level}"


def kind_task_name(seed: int, kind: str, size: int) -> str:
    return f"synth_s{seed}_{kind}_n{size}"


def task_names(seed: int = DEFAULT_SEED) -> list[str]:
    grid = [task_name(seed, size, level) for size, level in itertools.product(SIZES, LEVELS)]
    kinds = [kind_task_name(seed, k, size) for k, size in itertools.product(KINDS, KIND_SIZES)]
    return grid + kinds


def corpora(seed: int = DEFAULT_SEED) -> list[tuple[str, Path]]:
    """Every (collection, corpus directory) the tasks search, for setup to import."""
    base = [(collection_name(seed, size), corpus_dir(seed, size)) for size in SIZES]
    kinds = [
        (collection_name(seed, size, kind), corpus_dir(seed, size, kind))
        for kind, size in itertools.product(KINDS, KIND_SIZES)
    ]
    return base + kinds


# --- kinds: dials past size and wording ----------------------------------------------------
#
# The size/level grid can't separate haskie from grep on correctness - an agent with grep and
# enough turns always gets there. Each kind changes what finding the answer takes, holding the
# reference at level 1 (the runbook's own words) so the kind is the only new variable:
#
# - supersede: a later change notice overrides the target's port and timeout. Every runbook in
#   this kind says notices take precedence, so the agent is pointed at them - it has to look.
# - multihop: the answer (the owning team's on-call channel) is on a separate team page.
# - pdf: the runbooks exist only as PDFs, whose compressed text grep can't see.
# - absent: the target isn't in the corpus, only its near-miss decoys. Right answer: not found.

KINDS = ("supersede", "multihop", "pdf", "absent")
KIND_SIZES = (50, 500)
KIND_LEVEL = 1
# Opaque file-name tags for kinds whose documents differ from the base runbooks. "absent" reuses
# the base runbooks unchanged, so it reuses their names too.
KIND_TAGS = {"supersede": "a", "multihop": "b", "pdf": "c"}
SUPERSEDE_POINTER = (
    "Change notices published after this page take precedence over it until it is next updated."
)
MULTIHOP_POINTER = "The owning team's on-call channel is listed on its team directory page."
CHANNEL_WORDS = (
    "tiller", "lantern", "harbor", "quill", "ember", "cinder", "pylon", "gable", "rivet",
    "fathom", "keel", "sprocket",
)  # fmt: skip
MANAGERS = (
    "Ines Albrecht", "Tomas Varga", "Priya Nair", "Oren Castell", "Mei Lindqvist", "Dario Pell",
    "Aiko Brandt", "Felix Moreau", "Sanna Okafor", "Jonah Reyes", "Lotte Kovacs", "Yusuf Demir",
)  # fmt: skip


@dataclass(frozen=True)
class Notice:
    number: int  # drawn at random, so a notice's file name can't be paired with its runbook's
    port: int
    timeout_seconds: int


@dataclass(frozen=True)
class TeamPage:
    number: int
    channel: str
    manager: str


def kind_doc(seed: int, kind: str, number: int, suffix: str = ".md") -> str:
    return f"doc-s{seed}-{KIND_TAGS[kind]}-{number:04d}{suffix}"


def notices(seed: int, ordered: list[Service], target_index: int) -> dict[int, Notice]:
    """Change notices by corpus position: the target always has one, and so do about a third of
    the rest - enough that a notice existing says nothing about which service it's for."""
    rng = random.Random(f"{seed}-supersede")
    numbers = rng.sample(range(1000, 10000), len(ordered))
    ports = rng.sample(range(30000, 40000), len(ordered))
    found = {}
    for i, service in enumerate(ordered):
        if i == target_index or rng.random() < 0.3:
            timeout = rng.choice([t for t in TIMEOUTS if t != service.timeout_seconds])
            found[i] = Notice(numbers[i], ports[i], timeout)
    return found


def notice_text(service: Service, change: Notice) -> str:
    return (
        f"# Change notice CN-{change.number}\n\n"
        f"Applies to `{service.name}`. Effective immediately.\n\n"
        f"The listen port moves from {service.port} to {change.port}, and the request timeout "
        f"changes from {service.timeout_seconds} to {change.timeout_seconds} seconds. The "
        "service's runbook shows the old values until its next documentation pass; this notice "
        "takes precedence.\n"
    )


def team_pages(seed: int) -> dict[str, TeamPage]:
    rng = random.Random(f"{seed}-multihop")
    numbers = rng.sample(range(1000, 10000), len(TEAMS))
    words = rng.sample(CHANNEL_WORDS, len(TEAMS))
    managers = rng.sample(MANAGERS, len(TEAMS))
    return {
        team: TeamPage(numbers[i], f"#{words[i]}-{rng.randint(10, 99)}", managers[i])
        for i, team in enumerate(TEAMS)
    }


def team_text(team: str, page: TeamPage) -> str:
    return (
        f"# {team}\n\n"
        f"Engineering manager: {page.manager}.\n\n"
        f"On-call channel: {page.channel}. Page it for anything the team's services raise out "
        "of hours.\n"
    )


def to_pdf(text: str) -> bytes:
    """A minimal text PDF with a deflated content stream: grep sees compressed bytes, while any
    PDF text extractor reads it normally. Deterministic - no timestamps, no ids."""
    lines = [part for raw in text.splitlines() for part in (textwrap.wrap(raw, 95) or [""])]
    pages = [lines[i : i + 60] for i in range(0, len(lines), 60)]
    objects = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for page in pages:
        shown = "".join(f"({_pdf_escape(line)}) '\n" for line in page)
        stream = zlib.compress(f"BT /F1 10 Tf 12 TL 50 770 Td\n{shown}ET".encode("latin-1"))
        header = b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(stream)
        objects.append(header + stream + b"\nendstream")
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % len(objects)
        )
        kids.append(len(objects))
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    refs = b" ".join(b"%d 0 R" % kid for kid in kids)
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (refs, len(kids))
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % xref
    return out


def _pdf_escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


FIELDS = (
    "    PORT: int                     # the port it listens on\n"
    "    MAX_RETRY_ATTEMPTS: int       # the most attempts it makes on a failing call\n"
    "    REQUEST_TIMEOUT_SECONDS: int  # its request timeout, in seconds\n"
    "    OWNING_TEAM: str              # the team that owns it, spelled as the documentation "
    "spells it\n"
)
MULTIHOP_FIELDS = (
    "    PORT: int                     # the port it listens on\n"
    "    ON_CALL_CHANNEL: str          # the channel that pages the team that owns it\n"
)
KIND_PROMPT = """Write `answers.py` in the current directory. Standard library only.

Our internal documentation describes {reference}. Record the following, as currently in effect:

{fields}    FOUND: bool                   # False if the documentation describes no such service

If no such service is documented, set FOUND = False and every other name to None. None of these
can be inferred or guessed - they are specific to this organization.
"""


def _test_module(doc: str, cases: list[tuple[str, bool, str]]) -> str:
    """A test file from (name, discriminating, assertion) cases. `discriminating` marks what the
    kind is testing - the rest checks the run got the basics right on the way there."""
    body = "".join(
        f"\n\n{'@pytest.mark.discriminating' + chr(10) if discriminating else ''}"
        f"def test_{name}() -> None:\n    assert {assertion}\n"
        for name, discriminating, assertion in cases
    )
    header = f'"""Generated by `evals/synth.py` - do not edit.\n\n{doc}\n"""\n\n'
    return f"{header}import answers\nimport pytest\n{body}"


def _write_kind(
    seed: int, kind: str, size: int, ordered: list[Service], target_index: int, texts: list[str]
) -> None:
    directory = corpus_dir(seed, size, kind)
    directory.mkdir(parents=True)
    target = ordered[target_index]
    team = (target.team, target.team.removeprefix("Team "))
    facts = [
        ("found", False, "answers.FOUND is True"),
        ("port", True, f"answers.PORT == {target.port}"),
        ("max_retry_attempts", True, f"answers.MAX_RETRY_ATTEMPTS == {target.max_attempts}"),
        ("timeout", True, f"answers.REQUEST_TIMEOUT_SECONDS == {target.timeout_seconds}"),
        ("owning_team", True, f"answers.OWNING_TEAM in {team!r}"),
    ]
    fields = FIELDS

    def rng(service: Service) -> random.Random:
        """Each runbook's boilerplate comes from its own name, exactly as in the base corpus."""
        return random.Random(f"{seed}-{service.name}")

    if kind == "supersede":
        # Only the services kind corpora use - sampling over all of `ordered` would change every
        # notice whenever a larger size is added.
        changes = notices(seed, ordered[: max(KIND_SIZES)], target_index)
        for i in range(size):
            service = ordered[i]
            text = runbook(service, rng(service), SUPERSEDE_POINTER)
            (directory / kind_doc(seed, kind, i)).write_text(text)
            if i in changes:
                (directory / kind_doc(seed, kind, changes[i].number)).write_text(
                    notice_text(service, changes[i])
                )
        change = changes[target_index]
        evidence = [kind_doc(seed, kind, target_index), kind_doc(seed, kind, change.number)]
        # Discriminating: only what the notice overrides. The rest is the runbook's, unchanged.
        facts = [
            ("found", False, "answers.FOUND is True"),
            ("port", True, f"answers.PORT == {change.port}"),
            ("max_retry_attempts", False, f"answers.MAX_RETRY_ATTEMPTS == {target.max_attempts}"),
            ("timeout", True, f"answers.REQUEST_TIMEOUT_SECONDS == {change.timeout_seconds}"),
            ("owning_team", False, f"answers.OWNING_TEAM in {team!r}"),
        ]
        doc = f"Target `{target.name}`; the notice CN-{change.number} overrides port and timeout."
    elif kind == "multihop":
        pages = team_pages(seed)
        for i in range(size):
            service = ordered[i]
            text = runbook(service, rng(service), MULTIHOP_POINTER)
            (directory / kind_doc(seed, kind, i)).write_text(text)
        for name, page in pages.items():
            (directory / kind_doc(seed, kind, page.number)).write_text(team_text(name, page))
        page = pages[target.team]
        evidence = [kind_doc(seed, kind, target_index), kind_doc(seed, kind, page.number)]
        fields = MULTIHOP_FIELDS
        facts = [
            ("found", False, "answers.FOUND is True"),
            ("port", False, f"answers.PORT == {target.port}"),
            ("on_call_channel", True, f"answers.ON_CALL_CHANNEL == {page.channel!r}"),
        ]
        doc = f"Target `{target.name}`, owned by {target.team}; its channel is {page.channel}."
    elif kind == "pdf":
        for i in range(size):
            (directory / kind_doc(seed, kind, i, ".pdf")).write_bytes(to_pdf(texts[i]))
        evidence = [kind_doc(seed, kind, target_index, ".pdf")]
        doc = f"Target `{target.name}`, as a PDF only."
    else:  # absent
        for i in range(size):
            if i != target_index:
                (directory / doc_name(seed, i)).write_text(texts[i])
        evidence = []
        facts = [
            ("not_found", True, "answers.FOUND is False"),
            ("no_decoy_values", False, "answers.PORT is None"),
        ]
        doc = f"`{target.name}` is left out; only its look-alikes remain. The answer is not found."

    task = TASK_ROOT / kind_task_name(seed, kind, size)
    shutil.rmtree(task, ignore_errors=True)
    task.mkdir(parents=True)
    meta = {
        "module": "answers",
        "evidence": evidence,
        "corpus": str(directory.relative_to(ROOT)),
        "collection": collection_name(seed, size, kind),
    }
    (task / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    prompt = KIND_PROMPT.format(reference=reference(target, KIND_LEVEL), fields=fields)
    (task / "task.md").write_text(prompt)
    (task / "test_task.py").write_text(_test_module(doc, facts))


TASK_PROMPT = """Write `answers.py` in the current directory. Standard library only.

Our internal documentation describes {reference}. Record four facts about that service, exactly as
documented:

    PORT: int                     # the port it listens on
    MAX_RETRY_ATTEMPTS: int       # the most attempts it makes on a failing call
    REQUEST_TIMEOUT_SECONDS: int  # its request timeout, in seconds
    OWNING_TEAM: str              # the team that owns it, spelled as the documentation spells it

None of these can be inferred or guessed - they are specific to this organization.
"""

TEST_TEMPLATE = '''"""Generated by `evals/synth.py` - do not edit.

Seed {seed}, {size} documents, level {level}. The target is `{name}`, documented in `{doc}`.
Every assertion is discriminating: the values are drawn at random per seed, so only a run that
found the target's runbook can pass.
"""

import answers
import pytest


@pytest.mark.discriminating
def test_port() -> None:
    assert answers.PORT == {port}


@pytest.mark.discriminating
def test_max_retry_attempts() -> None:
    assert answers.MAX_RETRY_ATTEMPTS == {max_attempts}


@pytest.mark.discriminating
def test_request_timeout_seconds() -> None:
    assert answers.REQUEST_TIMEOUT_SECONDS == {timeout_seconds}


@pytest.mark.discriminating
def test_owning_team() -> None:
    """The runbook says "Owned by {team}": with or without the word "Team", it names one team."""
    assert answers.OWNING_TEAM in ({team!r}, {bare_team!r})
'''


def reference(service: Service, level: int) -> str:
    if level == 0:
        return f"a service called `{service.name}`"
    return f"the service that {describe(service.purpose, paraphrased=level == 2)}"


def extra_services(seed: int, base: list[Service], count: int) -> list[Service]:
    """Services past the first `BASE_POOL`, from their own random stream so the base corpus stays
    byte-for-byte what was already imported. Names come from new word lists; purposes may repeat
    among these distractors (there are fewer combinations than services) but never the target's,
    and a fifth are near misses of it, as in the base."""
    rng = random.Random(f"{seed}-extra")
    names = rng.sample([f"{f}-{s}" for f in FISH for s in STARS], count)
    taken = {service.port for service in base}
    ports = rng.sample([p for p in range(20000, 30000) if p not in taken], count)
    target = base[0].purpose
    dims = (len(ACTIONS), len(OBJECTS), len(COUNTERPARTIES), len(SCHEDULES))
    extras = []
    for i in range(count):
        purpose = target
        while purpose == target:
            if rng.random() < 0.2:
                axis = rng.randrange(4)
                changed = list(target)
                changed[axis] = rng.randrange(dims[axis])
                purpose = (changed[0], changed[1], changed[2], changed[3])
            else:
                action, obj, cp, schedule = (rng.randrange(d) for d in dims)
                purpose = (action, obj, cp, schedule)
        extras.append(
            Service(
                names[i],
                purpose,
                ports[i],
                rng.randint(2, 9),
                rng.choice(TIMEOUTS),
                rng.choice(TEAMS),
            )  # fmt: skip
        )
    return extras


def layout(seed: int) -> tuple[list[Service], int]:
    """Every service in corpus order, and the target's position in it. The target sits somewhere
    among the first five documents, so it is in every size."""
    everything = services(seed, BASE_POOL)
    target_index = random.Random(seed).randrange(5)
    order = list(range(1, BASE_POOL))
    order.insert(target_index, 0)
    ordered = [everything[i] for i in order]
    return ordered + extra_services(seed, everything, max(SIZES) - BASE_POOL), target_index


def generate(seed: int = DEFAULT_SEED) -> list[str]:
    """Write the corpora and the task directories; return the task names. Deterministic for a
    seed, and it replaces whatever an earlier run wrote for that seed."""
    _assert_paraphrases_disjoint()
    _assert_boilerplate_neutral()
    ordered, target_index = layout(seed)
    texts = [runbook(service, random.Random(f"{seed}-{service.name}")) for service in ordered]

    shutil.rmtree(SYNTH_DIR / f"seed-{seed}", ignore_errors=True)
    for size in SIZES:
        directory = corpus_dir(seed, size)
        directory.mkdir(parents=True)
        for index in range(size):
            (directory / doc_name(seed, index)).write_text(texts[index], encoding="utf-8")

    target = ordered[target_index]
    for size, level in itertools.product(SIZES, LEVELS):
        task = TASK_ROOT / task_name(seed, size, level)
        shutil.rmtree(task, ignore_errors=True)
        task.mkdir(parents=True)
        meta = {
            "module": "answers",
            "evidence": [doc_name(seed, target_index)],
            "corpus": str(corpus_dir(seed, size).relative_to(ROOT)),
            "collection": collection_name(seed, size),
        }
        (task / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (task / "task.md").write_text(TASK_PROMPT.format(reference=reference(target, level)))
        (task / "test_task.py").write_text(
            TEST_TEMPLATE.format(
                seed=seed,
                size=size,
                level=level,
                name=target.name,
                doc=doc_name(seed, target_index),
                port=target.port,
                max_attempts=target.max_attempts,
                timeout_seconds=target.timeout_seconds,
                team=target.team,
                bare_team=target.team.removeprefix("Team "),
            )
        )
    for kind, size in itertools.product(KINDS, KIND_SIZES):
        _write_kind(seed, kind, size, ordered, target_index, texts)
    return task_names(seed)


if __name__ == "__main__":
    for seed in SEEDS:
        for task in generate(seed):
            print(task)
