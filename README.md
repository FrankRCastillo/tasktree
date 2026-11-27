# tasktree

A strictly local, recursive task harness that lets small LLMs complete projects far larger than any single call could hold: long documents, multi-file programs, analyses of an existing repo.

It works like a chain of command. Every node in the tree is an agent that either does its task itself or splits it among subordinates. Context flows down, so each worker knows the broad intent and what its peers are doing. Reports flow up, so each manager judges results without reading everything. No call ever has to see the whole project.

## How it works

Each node makes one `agent` call and picks a mode:

- **worker**: the task fits in one focused pass (roughly 1,500 words of prose, or one file or module of code). The worker writes the complete deliverable and a short summary for its manager.
- **manager**: the task is too big. The manager splits it into ordered, non-overlapping subtasks, divides any budget (such as a word count) among them, and can set `shared_invariants` (tone, audience, conventions, schemas) that its whole branch must follow.

**Context flows down.** A worker's prompt includes:

- **Chain of Command**: the intent and success criteria of every node above it, root first
- **Shared Invariants**: the rules set by any of its managers
- **Sibling Overview**: its peers' scopes and status, plus the reports of peers that have finished
- **End of the Preceding Section**: the tail of the content that will sit directly before its own, even across branches, so its work can pick up where that leaves off
- **Dependencies**: interface stubs (for Python files) or summaries of the siblings it builds on

**Reports flow up.** When a manager's subordinates finish, the manager reviews their reports (summary, `wc -w` word count, verify-command result), not their full text. It either approves or sends targeted revisions to specific subordinates, for up to `max_review_rounds` rounds. A revision sent to a sub-manager goes to that sub-manager's own review, which passes it further down.

**Managers don't rewrite.** A manager's deliverable is its children's content joined in order. Children that write their own file (`file_path`) deliver it to the workspace instead. Coherence comes from the context each worker gets, not from a final model call re-writing everything.

## Install

```sh
pip install -e .            # or: pip install -e '.[dev]' for pytest
```

You need Python 3.10+ and an OpenAI-compatible chat endpoint that supports `response_format={"type": "json_object"}`, such as [llama.cpp's `llama-server`](https://github.com/ggml-org/llama.cpp) or [Ollama](https://ollama.com).

> **Context window:** prompts carry the chain of command, peer reports and reference files, so give the server plenty of context. Ollama's default context window is small and is not configurable per request through its OpenAI-compatible API, so set `num_ctx` in a Modelfile or set `OLLAMA_CONTEXT_LENGTH`. For `llama-server`, use `-c`. Leave room for `max_tokens` of output on top of the prompt.

## Usage

```sh
tasktree --init-config                      # write ~/.tasktreerc with the defaults, then edit it

tasktree "Write a 5,000-word field guide to North Atlantic seabirds" -o guide.md
tasktree "Build a CLI todo app in Python with tests" -w ./todo
tasktree "Document the architecture of this codebase" -c ~/src/some-repo -o ARCHITECTURE.md

tasktree --resume -w ./todo                 # continue an interrupted or partly failed run
```

| Flag | Meaning |
|---|---|
| `-o, --output FILE` | Save the root's aggregated output to `FILE`. |
| `-w, --workspace DIR` | Where files are written (default `./tasktree_workspace`). |
| `-c, --context DIR` | Read-only directory the agents can list and read (`read_files`). |
| `--rc FILE` | Config file (default `~/.tasktreerc`). |
| `--init-config` | Write the default config to `--rc`, overwriting it. |
| `--resume` | Resume the run checkpointed in the workspace. |

The live tree shows each node's status: `PENDING`, `WORKING`, `REVISING`, `TESTING`, `MANAGING`, `REVIEWING`, `DONE` or `FAILED`. Active calls show a token counter, which also counts a thinking model's hidden reasoning.

### The workspace

- Files written by workers, and by managers that have a `file_path`, live here. Model-chosen paths are confined to the workspace.
- It's a git repository with one commit per completed node, so you can diff or roll back each step.
- `.tasktree_state.json` is the checkpoint. `--resume` keeps every `DONE` node and every manager's split, and re-runs everything else. A node's ancestors re-run too, since their output was built from incomplete work.

## Configuration

`~/.tasktreerc` is an INI file layered over the [built-in defaults](src/tasktree/default_tasktreerc), so it only needs the keys you want to change. tasktree warns about any key it doesn't recognize.

**`[model]`**

| Key | Default | |
|---|---|---|
| `base_url` | `http://localhost:11434/v1` | OpenAI-compatible endpoint (Ollama by default) |
| `api_key` | `ollama` | Sent as-is; local servers usually ignore it |
| `model` | `qwen2.5:14b` | Model name to request |
| `temperature` | `0.2` | |
| `max_tokens` | `8192` | Per call; a worker's whole deliverable must fit, plus its JSON |
| `enable_thinking` | `true` | `false` asks thinking models (e.g. Qwen3 on llama.cpp) to skip hidden reasoning via `chat_template_kwargs`: much faster, possibly less careful. `true` sends nothing extra, so the server's default applies |

**`[harness]`**

| Key | Default | |
|---|---|---|
| `max_depth` | `3` | Nodes at this depth must work rather than split |
| `max_children` | `4` | Most subtasks a manager may create |
| `max_retries` | `3` | Attempts per worker (invalid output, bad path, failed verify) |
| `max_review_rounds` | `2` | Reviews per manager; revisions happen between rounds. `0` approves without review |
| `workspace` | `./tasktree_workspace` | |

**`[prompts]`**

- **`agent`**: the system prompt for every node. `{max_children}` is replaced with the config value; any other braces are left alone.
- **`manager_review`**: the system prompt for a manager's review.

The JSON schema each call must match is added automatically, and the structured context sections described above are part of the user message.

## Safety

- Workers may set a `verify_command`, which runs **through your shell** in the workspace, with a 60-second timeout.
- The commands come from model output, so use tasktree only with models and tasks you trust, or run it inside a container or VM.
- File writes are confined to the workspace, and reads to the `-c` directory.

## Development

```sh
pytest
```

The tests drive the harness with a scripted fake client, so they need no model server.
