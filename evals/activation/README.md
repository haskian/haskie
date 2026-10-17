# haskie activation eval

Does Claude reach for haskie when a prompt never mentions it, and what makes it? Activation only:
no task to finish and no grader. Each run stops after a few turns and costs cents.

`prompts.jsonl` holds 20 plain user requests:

- 12 should reach for the library: topics the book collection covers. Some are ones Claude
  believes it knows, which is the hard case.
- 8 are controls haskie's own rule says to leave alone: arithmetic, trivia, a poem, a regex, code
  in front of it.

Each prompt runs under each setup:

| setup | what the agent gets beyond haskie's MCP tools |
| --- | --- |
| `none` | nothing |
| `rule` | the rule and skill `haskie install claude` writes, rendered by haskie's own code |
| `hook` | a `UserPromptSubmit` hook that injects haskie's top passages (`evals/steer.py`) |
| `rule+hook` | both |

The report gives, per setup, the share of should-trigger prompts where the library was consulted
(Claude searched it, or the hook injected passages), the share of controls where it was (false
activations), and the cost. `searched` counts Claude's own searches apart from the hook's.

```sh
mise run eval:activation                                     # every setup, 2 samples, sonnet
mise run eval:activation -- --setups none rule --samples 3
mise run eval:activation -- --model haiku --only raft-log math
```

Runs are isolated like the agent eval's: no auto memory, a working directory outside the home,
denied reads, Bash sandboxed, and searches scoped to the book collection (`evals/scope.py`). It
needs the full-text eval instance (`mise run eval:setup`), and writes `reports/<timestamp>/`.
