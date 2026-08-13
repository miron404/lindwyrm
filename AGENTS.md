# Project notes for coding agents

## What this is

lindwyrm is a minimal CLI coding agent that talks to Anthropic- and
OpenAI-compatible model APIs directly over HTTP. It ships as `lindwyrm` on
PyPI; the command is `lwyrm`. Two runtime dependencies, no telemetry, and
small enough to read in an afternoon — that last property is a design
constraint, not an accident. Weigh new dependencies and new abstraction
layers against it.

## Commands

```bash
python -m unittest discover -s tests   # the whole suite, ~1s
python -m unittest tests.test_offload  # one module
pip install -e .                       # dev install
pip install -e '.[socks]'              # with SOCKS proxy support
python -m build                        # build both distributions
```

## Layout

Only what isn't obvious from the names:

- `http.py` — shared by both wire clients: pooled `httpx.Client` (one per
  proxy setting), retry/backoff, SSE parsing, proxy routing.
- `client.py` / `openai_client.py` — the two wire formats. Everything else in
  the codebase speaks one internal representation (Anthropic-style content
  blocks); `openai_client.py` is a translator at the edges only.
- `offload.py` — moves bulky tool results to disk, leaving a stub.
- `project.py` — loads this file into the system prompt.
- `packaging/lwyrm/` — alias distribution so `pip install lwyrm` works. No
  code; its version and dependency floor are stamped by CI.

## Conventions

- Tests are stdlib `unittest`. Do not add pytest or any other test
  dependency — zero test dependencies is a hard rule.
- No provider SDKs. `httpx` and the documented wire format, nothing else.
  Avoiding a long supply chain around API keys is a founding reason for the
  project.
- Comments explain *why*, not what. If a line looks odd, the comment should
  say what breaks without it.
- Every new `Preset` field must also be added to `PRESET_FIELDS` in `cli.py`.
- New behaviour comes with tests. Anything reachable without a network call
  is expected to be covered.

## Gotchas

Each of these has already cost someone an afternoon:

- **Version lives only in `lindwyrm/__init__.py`.** setuptools reads it as
  dynamic metadata and the publish workflow stamps the same value into the
  alias package. Never hand-edit `packaging/lwyrm/pyproject.toml`'s version.
- **Tag the commit that bumps the version, and bump it last.** A tag that
  points a commit earlier builds the wrong tree. The workflow now fails on a
  mismatch; before that check existed it published a release with no license.
- **A version on PyPI can never be re-uploaded.** Mistakes are fixed by
  releasing forward, never by moving a tag.
- **DeepSeek's Anthropic endpoint returns 400 unless thinking blocks are
  echoed back** in message history. This is why assistant content is stored
  and replayed verbatim.
- **History may only be cut at a user-turn boundary.** Cutting elsewhere
  orphans a `tool_result` from its `tool_use` and both APIs reject it. See
  `is_turn_boundary`.
- **The system prompt and tool schemas are the cacheable prefix.** Keep them
  stable and keep history append-only. Never put volatile data (timestamps,
  git state) at the front of the system prompt — it invalidates the cache on
  every single turn.
- **Environment proxies are ignored unless `proxy = "system"`.** Clients are
  built with `trust_env=False` on purpose; without it a stray `ALL_PROXY`
  reroutes API traffic and an explicit "go direct" setting isn't direct.
- **`localhost` is never proxied.** A proxy resolves it on its own side, so a
  proxied request for a local model server reaches a stranger's loopback.

## Out of scope

Leave these alone unless asked directly:

- `.github/workflows/publish.yml` — publishing credentials flow through it.
- Anything that widens the dependency list.
