# lindwyrm

A minimal, dependency-light coding agent that talks directly to model APIs --
**DeepSeek** (V4 Flash / Pro) out of the box, plus any other **Anthropic- or
OpenAI-compatible** provider you add as a preset (OpenRouter-style
aggregators, Kimi, a local vLLM/Ollama server, whatever you use).

Built because the off-the-shelf agents carry telemetry and a long supply chain
of packages that have a habit of trying to exfiltrate API keys. This is one
small package with a single runtime dependency (`httpx`, plus optional `rich`
for markdown rendering). You can read all of it in an afternoon.

## What it does

- **Presets**: DeepSeek Flash/Pro ship built in (`flash`/`pro`). Add your own
  via `[[presets]]` in config to point at any other provider -- each preset
  bundles a wire format (`anthropic` or `openai`), `base_url`, `model`, and
  which env var(s) hold its key. Switch anytime with `/model <name>` or
  `-m <name>`; `/presets` lists what's available.
- Talks to the API directly over raw HTTP (no `openai`/`anthropic` SDK) --
  streams responses including **thinking/reasoning** where supported, and
  correctly echoes DeepSeek's thinking blocks back in history (its Anthropic
  endpoint returns 400 otherwise). For OpenAI-format presets, reasoning is
  shown live if the provider streams it but isn't echoed back (not required
  there, and some providers reject unknown fields).
- Tools: `read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`,
  `delete_file`, `bash` -- work identically no matter which preset/provider
  is active. `edit_file` requires a unique match by default and names the
  lines that matched when it isn't unique, or takes `replace_all`. `grep`
  streams files line by line and skips binaries and vendored directories.
- **Path-based permissions**: every read/write/delete resolves to a level —
  `allow` (silent), `confirm` (ask first), or `deny` (blocked) — from a global
  default plus per-path rules. The most specific rule wins, so you can open a
  folder broadly and still lock a subfolder inside it. Defaults: read=allow,
  write=confirm, delete=confirm. `../` and symlink escapes are resolved before
  matching, so a link can't dodge a rule.
- **Set rules in config or on the fly**: `[[policy.rules]]` entries in your
  `.lindwyrm.toml`, or `/perm <path> write=allow delete=confirm` during a session.
- **Bash is separate** (a shell command isn't path-bound): a single global level
  (default confirm) plus an always-on denylist and an optional allowlist for
  safe commands (`git status`, `ls`, …).
- Optional **read-only** mode (forces write/delete to deny) and a JSON-lines
  **audit log**.
- **Markdown rendering** of answers via `rich` (code blocks with syntax
  highlighting, tables, lists), streamed live as the model writes.
- **Reasoning display** you control: thinking streams live and then vanishes
  when the answer begins (`peek`), or stays in scrollback (`show`), or is
  hidden (`hide`). `/think` reprints the last turn's reasoning on demand.
- **Context management**: every token of history is re-sent (and re-billed) on
  every turn, so long sessions get expensive well before they hit the model's
  limit. lindwyrm tracks the token count the API actually reports and, once the
  window is 75% full, summarizes older history away and continues. `/context`
  shows how full it is; `/compact` does it on demand.
- **Retries**: 429s and 5xx are retried with exponential backoff and jitter
  (honoring `Retry-After`), so one blip doesn't kill a turn's work. A request
  is only retried if nothing has been streamed yet — never mid-answer.
- No telemetry. API keys are read only from the environment or a key file you
  point at (per preset) — never stored inline in a committed config.

## Install

```bash
pip install lindwyrm       # installs the `lwyrm` command (and `lindwyrm` as an alias)
export DEEPSEEK_API_KEY=sk-...
```

`pip install lwyrm` works too — it's an alias package that pulls in the same
thing, for when you reach for the command's name first.

From a checkout instead:

```bash
git clone https://github.com/miron404/lindwyrm && cd lindwyrm
pip install -e .
```

(Requires Python 3.11+.)

## Use

```bash
lwyrm                    # interactive REPL in the current directory
lwyrm -m pro             # use the deepseek-pro preset
lwyrm -m kimi            # use a custom preset you defined in config
lwyrm --no-thinking      # disable thinking mode
lwyrm --read-only        # answer only; never write or run
lwyrm -C ~/myproject     # set project root
lwyrm -p "add type hints to utils.py"   # one-shot, then exit
```

Slash commands in the REPL: `/model <name>` (switch preset), `/presets` (list
them), `/thinking on|off`, `/think peek|show|hide` (or `/think` to reprint last
reasoning), `/markdown on|off`, `/perm <path> read=.. write=.. delete=..` (set
per-path rules; `reset` clears; no args shows the table), `/policy`,
`/context` (how full the window is), `/compact` (summarize history now),
`/clear`, `/help`, `/exit`.

When the agent wants to write, delete, or run a command, you'll see a prompt:

```
  WRITE requested: create src/main.py (412 chars)
    print("hello")
    ...
  Allow? [y]es / [n]o / [a]lways (write) / [q]uit:
```

`always` grants that operation for the rest of the current user turn only.

## Configure

Copy `lindwyrm.example.toml` to `./.lindwyrm.toml` (project) or
`~/.config/lindwyrm/config.toml` (user). Project settings override user settings.
Key options:

```toml
default_preset = "flash"   # "flash", "pro", or a name from [[presets]] below

# Add your own provider (Anthropic- or OpenAI-compatible):
[[presets]]
name = "kimi"
format = "openai"
base_url = "https://api.tokenrouter.com/v1"
model = "moonshotai/kimi-k3-free"
api_key_env = ["TOKENROUTER_API_KEY"]
thinking = false

[policy]
read   = "allow"        # global defaults; most specific rule below wins
write  = "confirm"
delete = "confirm"
bash   = "confirm"      # bash isn't path-scoped
bash_allowlist = ["git status", "ls"]
bash_denylist  = ["sudo", "curl"]

[[policy.rules]]        # write freely in generated/, but still confirm deletes
path = "generated"
write = "allow"

[[policy.rules]]        # keep the agent out of your keys
path = "~/.ssh"
read = "deny"
```

## Security notes & limits

- `bash` runs through the shell. The denylist is a backstop, not a jail — the
  real protection is that `bash` defaults to `confirm`, so you see every command
  before it runs. If you want stronger isolation, run lindwyrm inside a container.
- Path permissions protect against accidental/strayed file operations and
  resolve symlinks/`../` before matching; they are not a defense against a write
  you confirm yourself.
- Compaction is lossy by nature: it replaces older turns with a model-written
  summary. It cuts only at user-turn boundaries so tool calls are never split
  from their results, but details outside the summary are gone. Use `/clear`
  between unrelated tasks rather than relying on it.

## Tests

```bash
python -m unittest discover -s tests
```

Stdlib `unittest`, no test dependencies. Covers permission resolution, path
escaping (`../` and symlinks), the OpenAI wire-format translation, compaction
boundaries, and retry backoff — everything that can be checked without a
network call.

## Releasing

Publishing runs on tag push via GitHub Actions using PyPI **Trusted
Publishing** (OIDC) — no API token is stored in this repo. One tag publishes
two distributions: `lindwyrm` (the package) and `lwyrm` (an alias metapackage
that only depends on it, so the name can't be typosquatted).

```bash
# bump __version__ in lindwyrm/__init__.py, then:
git tag v0.1.0 && git push origin v0.1.0
```

`lindwyrm/__init__.py` is the single source of truth for the version: the main
package reads it via setuptools' dynamic metadata, and the workflow stamps the
same value into the alias package before building it.

Both projects publish from the same `pypi` environment. Note for anyone setting
this up from scratch: a *pending* trusted publisher must be unique across
owner + repository + workflow + environment, so the second project cannot be
pre-registered against the same environment while the first is still pending.
Once the first project has actually been published its publisher is no longer
pending and the second can be registered normally — or give each project its
own environment from the start.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 miron404.
