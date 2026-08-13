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

- **Project instructions**: drop an `AGENTS.md` in the repo root (or run
  `/init` for a starter) and every session begins knowing your build commands,
  conventions and gotchas instead of rediscovering them. A personal
  `~/.config/lindwyrm/AGENTS.md` applies everywhere. Both are loaded as
  reference material, explicitly not as instructions that outrank you — a file
  that arrives with someone else's clone shouldn't be able to give orders to an
  agent that can run commands.
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
  highlighting, tables, lists). The answer is printed block by block as it
  completes — full width, scrolling like any other program's output — rather
  than being redrawn inside a fixed region, which cannot scroll and drags the
  terminal back down on every repaint.
- **Reasoning display** you control: thinking streams live and then vanishes
  when the answer begins (`peek`), or stays in scrollback (`show`), or is
  hidden behind a spinner (`hide`). `/think` reprints the last turn's
  reasoning on demand, whichever mode you were in.
- **Context management**: every token of history is re-sent (and re-billed) on
  every turn, so long sessions get expensive well before they hit the model's
  limit. lindwyrm tracks the token count the API actually reports and, once the
  window is 75% full, reclaims space in two stages. First it **offloads** bulky
  tool results to disk, leaving a stub the model can expand with
  `read_offloaded` — cheap and reversible. Only if that isn't enough does it
  **summarize** older history, which is lossy. `/context` shows how full the
  window is (and how much is being served from cache); `/compact [instructions]`
  runs it on demand, optionally told what to keep.
- **Retries**: 429s and 5xx are retried with exponential backoff and jitter
  (honoring `Retry-After`), so one blip doesn't kill a turn's work. A request
  is only retried if nothing has been streamed yet — never mid-answer.
- **Per-provider proxy**: a global `proxy` plus a per-preset override, including
  an explicit bypass so one provider can go direct while everything else is
  proxied. SOCKS5 (`pip install 'lindwyrm[socks]'`) or HTTP. Off by default,
  and `HTTP_PROXY`/`ALL_PROXY` are ignored unless you set `proxy = "system"` —
  a stray variable in your shell won't silently reroute API traffic.
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
lwyrm --continue         # pick up the last session in this project
lwyrm --resume <id>      # pick up a specific one (see /sessions)
lwyrm --no-save          # leave no trace on disk
```

Slash commands in the REPL: `/model <name>` (switch preset), `/presets` (list
them), `/thinking on|off`, `/think peek|show|hide` (or `/think` to reprint last
reasoning), `/markdown on|off`, `/perm <path> read=.. write=.. delete=..` (set
per-path rules; `reset` clears; no args shows the table), `/policy`,
`/context` (how full the window is), `/compact [instructions]` (compact now),
`/sessions` (list saved sessions), `/init` (write a starter `AGENTS.md`),
`/proxy` (show the proxy in use), `/clear`, `/help`, `/exit`.

When the agent wants to write, delete, or run a command, you'll see a prompt:

```
  WRITE requested: create src/main.py (412 chars)
    print("hello")
    ...
  Allow? [y]es / [n]o / [a]lways (write) / [q]uit:
```

`always` grants that operation for the rest of the current user turn only.

**Ctrl+C** stops whatever is running and clears the line you were typing, the
way a shell does; press it twice in a row, or use **Ctrl+D** or `/exit`, to
leave.

## Project instructions

`AGENTS.md` in the project root is read at the start of every session and
prepended to the system prompt. `/init` writes a commented starter template.
Keep it short and factual — it costs tokens on every single turn:

```markdown
## Commands
```bash
python -m unittest discover -s tests   # tests
ruff check .                           # lint
```

## Conventions
- stdlib `unittest`; never add pytest (zero test dependencies is a hard rule)
- line length 88

## Gotchas
- `packaging/lwyrm/` version is stamped by CI; don't hand-edit it


- Rules of thumb: write down anything you catch yourself repeating in chat, and
leave out anything the agent can discover in two seconds by looking. Commands
belong here verbatim so they aren't guessed. So do decisions it cannot infer
from the code — "we chose X over Y because Z" stops it from helpfully
reintroducing Y.

- `LINDWYRM.md` takes precedence if you want notes specific to this agent, and
`~/.config/lindwyrm/AGENTS.md` holds preferences that follow you across
projects. `context_file` in config points somewhere else entirely.

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

Proxies are opt-in and resolve most-specific-first — `--proxy`, then the
preset's own setting, then the global one:

```toml
proxy = "socks5h://127.0.0.1:1080"   # everything goes through here...

[[presets]]
name = "local-llm"
base_url = "http://localhost:11434/v1"
proxy = false                        # ...except this one, which goes direct
```

`localhost` and `127.0.0.0/8` are always reached directly, including under
`--proxy`, which otherwise overrides everything — a proxy resolves `localhost`
on its own side, so proxying a local model server would send the request to a
stranger's machine rather than yours. For a model server elsewhere on the LAN,
list it (exact host, domain suffix or CIDR):

```toml
no_proxy = ["192.168.0.0/16", ".internal", "ollama.box"]
```

`proxy = "system"` opts back in to `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`.
`socks5h://` and `socks5://` behave the same: httpx passes the hostname to the
proxy, so DNS is resolved on the proxy side either way.

## Security notes & limits

- `bash` runs through the shell. The denylist is a backstop, not a jail — the
  real protection is that `bash` defaults to `confirm`, so you see every command
  before it runs. If you want stronger isolation, run lindwyrm inside a container.
- Path permissions protect against accidental/strayed file operations and
  resolve symlinks/`../` before matching; they are not a defense against a write
  you confirm yourself.
- Compaction is lossy by nature: it replaces older turns with a model-written
  summary. It cuts only at user-turn boundaries so tool calls are never split
  from their results, but details outside the summary are gone. Offloading,
  which runs first, is not lossy — the full text stays on disk under
  `~/.local/share/lindwyrm/offload/` and is swept after 7 days. Note that an
  offloaded result is a *snapshot*: the file may have changed since, which is
  exactly why it is copied rather than re-read on demand.

## Sessions

Every turn is written to `~/.local/share/lindwyrm/sessions/`, so closing the
terminal doesn't throw away the conversation. `--continue` resumes the most
recent session **for the current project** — sessions are scoped to the
directory they ran in, so one checkout never resumes another's work.
`/sessions` lists them, `--resume <id>` picks one, `/clear` starts a new one
without touching what is already saved.

Resuming restores the offloaded tool results too, so the `[offloaded: ...]`
markers in the history still resolve. Files are written `0600` in a `0700`
directory: a session holds whatever the agent read, which is often source
code and sometimes more. `--no-save` skips writing entirely, and sessions
older than 30 days are swept.

## Cost

Add prices to a preset and lindwyrm keeps a running total: a dim line after
each turn, and a breakdown in `/context`. Prices are per million tokens:

```toml
[[presets]]
name = "flash"
price_input = 0.28        # fresh input
price_cache_read = 0.028  # cached input, usually far cheaper
price_output = 0.42
```

Cache reads, cache writes and fresh input are counted separately, because
they are billed separately — on a typical session most of the input is served
from cache, and lumping them together would overstate the bill several times
over. No prices ship built in: they change, and a stale number is worse than
none.

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

Bump the version in its own commit, made last, and tag that commit -- the
workflow refuses to build when the tag doesn't match the version in the code.
Without that check the mismatch is silent: `skip-existing` makes the upload of
an already-published version succeed, so the run goes green having published
nothing.

Both projects publish from the same `pypi` environment. Note for anyone setting
this up from scratch: a *pending* trusted publisher must be unique across
owner + repository + workflow + environment, so the second project cannot be
pre-registered against the same environment while the first is still pending.
Once the first project has actually been published its publisher is no longer
pending and the second can be registered normally — or give each project its
own environment from the start.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 miron404.
