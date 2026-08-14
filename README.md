# lindwyrm

[![tests](https://github.com/miron404/lindwyrm/actions/workflows/tests.yml/badge.svg)](https://github.com/miron404/lindwyrm/actions/workflows/tests.yml)
[![pypi](https://img.shields.io/pypi/v/lindwyrm)](https://pypi.org/project/lindwyrm/)
[![python](https://img.shields.io/pypi/pyversions/lindwyrm)](https://pypi.org/project/lindwyrm/)
[![license](https://img.shields.io/pypi/l/lindwyrm)](LICENSE)

A small, dependency-light coding agent for the terminal. It talks to model
APIs directly over HTTP — **DeepSeek** out of the box, plus any other
**Anthropic- or OpenAI-compatible** provider you add as a preset: an
aggregator, Kimi, a local vLLM or Ollama server, whatever you use.

Built because the off-the-shelf agents carry telemetry and a long supply
chain of packages, some of which have a habit of reaching for API keys. This
is one package with two runtime dependencies — `httpx` for the wire and
`rich` for rendering — and no SDKs. You can read all of it in an afternoon.

```
$ lwyrm
lindwyrm - coding agent
  preset:   deepseek-flash (anthropic, deepseek-v4-flash)
  thinking: on
  root:     ~/work/shop
  perms:    read=allow write=confirm delete=confirm bash=confirm
  /help for commands, /exit to quit

you › rewrite total() in prices.py as a one-line sum() expression

→ read_file prices.py
  ok 1  def total(items):

→ edit_file prices.py

  WRITE requested: edit prices.py
    prices.py: +1 -4
    @@ -1,5 +1,2 @@
     def total(items):
    -    result = 0
    -    for item in items:
    -        result = result + item["price"] * item["qty"]
    -    return result
    +    return sum(item["price"] * item["qty"] for item in items)
  Allow? [y]es / [n]o / [a]lways (write) / [q]uit: y
  ok Edited prices.py

Rewrote total() as a one-line sum() expression, keeping identical behaviour.
```

## Install

```bash
pip install lindwyrm        # installs the `lwyrm` command (`lindwyrm` also works)
export DEEPSEEK_API_KEY=sk-...
lwyrm                       # start in the current directory
```

`pip install 'lindwyrm[socks]'` adds SOCKS proxy support.
`pip install lwyrm` works too — an alias package that pulls in the same thing.

Requires Python 3.11+ on Linux or macOS. Windows works under WSL; a native
Windows shell won't, because the `bash` tool relies on POSIX process groups
to kill a command's whole process tree.

## Use

```bash
lwyrm                    # interactive REPL in the current directory
lwyrm -m pro             # use the deepseek-pro preset
lwyrm -m kimi            # use a custom preset from your config
lwyrm -C ~/myproject     # set the project root
lwyrm -p "add type hints to utils.py"   # one-shot, then exit
lwyrm --continue         # pick up the last session in this project
lwyrm --resume <id>      # pick up a specific one (see /sessions)
lwyrm --read-only        # answer only; never write or run
lwyrm --no-thinking      # turn reasoning off
lwyrm --no-save          # leave no trace on disk
lwyrm --init             # write a starter AGENTS.md and exit (no API key needed)
```

`lwyrm --help` lists the rest.

In the REPL: `/model` `/presets` `/thinking` `/think` `/markdown` `/perm`
`/policy` `/context` `/compact` `/sessions` `/init` `/proxy` `/clear`
`/help` `/exit`. `/help` explains each one.

**Ctrl+C** stops whatever is running and clears the line you were typing, the
way a shell does. Press it twice, or use **Ctrl+D** or `/exit`, to leave.

## Contents

[Tools](#tools) · [Permissions](#permissions) ·
[Project instructions](#project-instructions) ·
[Context and cost](#context-and-cost) · [Sessions](#sessions) ·
[Proxies](#proxies) · [Configuration](#configuration) ·
[Limits](#security-notes-and-limits) · [Tests](#tests) ·
[Releasing](#releasing)

## Tools

`read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`,
`delete_file`, `bash`, `read_offloaded` — identical no matter which
provider is active.

- `bash` streams output line by line as the command runs, with stdout and
  stderr interleaved in the order they actually happened, and kills the whole
  process tree on timeout or Ctrl+C.
- `edit_file` requires a unique match by default; when the text appears
  several times the error names the lines that matched, so the next attempt
  has something to go on. `replace_all` changes every occurrence.
- `grep` reads files line by line and skips binaries and vendored
  directories. `list_dir` shows dotfiles and marks symlinks with their target.
- Every write and every overwrite is previewed as a real unified diff.

Reasoning is streamed where the provider supports it. DeepSeek's Anthropic
endpoint returns 400 unless thinking blocks are echoed back in history, so
assistant content is stored and replayed verbatim. OpenAI-format providers
don't require that, and some reject unknown fields, so there reasoning is
shown live but not sent back.

## Permissions

Every read, write and delete resolves to `allow` (silent), `confirm` (ask
first) or `deny` (blocked), from a global default plus per-path rules. The
most specific rule wins, so you can open a folder broadly and still lock a
subfolder inside it. Defaults: read=allow, write=confirm, delete=confirm.

`../` and symlinks are resolved **before** matching, so a link can't dodge a
rule — and the confirmation prompt shows the real destination, not the path
that was typed.

`bash` sits outside this system, because a shell command isn't bound to a
path: it has one global level (default confirm), an always-on denylist and an
optional allowlist for harmless commands. Set rules in config under
`[[policy.rules]]`, or during a session with
`/perm src write=allow delete=confirm`.

There is also a `--read-only` mode and a JSON-lines audit log.

## Project instructions

`AGENTS.md` in the project root is read at the start of every session and
prepended to the system prompt, so the agent begins knowing your build
commands and conventions instead of rediscovering them. `/init` or
`lwyrm --init` writes a commented starter template.

Keep it short and factual — it costs tokens on every turn. Write down what
you catch yourself repeating in chat; leave out anything the agent can
discover in two seconds by looking:

````markdown
## Commands
```bash
python -m unittest discover -s tests   # tests
ruff check .                           # lint
```

## Conventions
- stdlib `unittest`; never add pytest (zero test dependencies is a hard rule)
- line length 88

## Gotchas
- the version lives only in `lindwyrm/__init__.py`; CI stamps the rest
````

Commands belong there verbatim so they aren't guessed, and so do decisions
the agent can't infer from the code — "we chose X over Y because Z" is what
stops it from helpfully reintroducing Y.

`LINDWYRM.md` takes precedence for notes specific to this agent, and
`~/.config/lindwyrm/AGENTS.md` holds preferences that follow you between
projects. Both are loaded as reference material, explicitly not as
instructions that outrank you: a file that arrives with someone else's clone
shouldn't be able to give orders to an agent that can run commands.

## Context and cost

Every token of history is re-sent, and re-billed, on every turn — so a long
session gets expensive well before it reaches the model's limit. lindwyrm
tracks the token count the API actually reports and, once the window is 75%
full, reclaims space in two stages:

1. **Offloading** moves bulky tool results to disk, leaving a short stub the
   model can expand with `read_offloaded`. Cheap, and reversible.
2. **Summarizing** older history, which is lossy, and only if the first stage
   wasn't enough.

`/context` shows how full the window is and how much input is being served
from cache; `/compact [instructions]` runs it on demand, optionally told what
to keep.

The built-in DeepSeek presets declare the real 1M window, so in practice the
trigger is a long way off and summarizing rarely happens at all — which is
the point, since it is the lossy stage. A single oversized tool result still
goes to disk immediately, regardless of how full the window is.

One provider quirk worth knowing: DeepSeek accepts `thinking_budget` and
ignores it, so on DeepSeek `max_tokens` is the only thing bounding how long
the model reasons — a hard question can spend the entire allowance thinking
before it starts to answer. Anthropic's own API honors the budget.

Add prices to a preset and a running total appears after each turn:

```toml
[[presets]]
name = "flash"
price_input = 0.44        # per million tokens, cache miss
price_cache_read = 0.014  # cache hit — around 30x cheaper
price_output = 1.32
```

Fresh input, cache reads and cache writes are counted separately, because
they are billed separately — on a typical session most of the input comes
from cache, and lumping them together overstates the bill several times over.
No prices ship built in: they change — DeepSeek's moved twice while this was
being written, and now vary by time of day. `lindwyrm.example.toml` carries
the current published figures with the date they were taken; they are the
peak ones, since overstating is the safer error. Time-of-day pricing is not
modelled, so halve them if you work off-peak.

That ~30x gap between a hit and a miss is why the system prompt and tool
schemas are kept byte-stable across turns: anything that shifts the start of
the prompt re-charges the whole conversation at miss rates.

## Sessions

Every turn is written to `~/.local/share/lindwyrm/sessions/`, so closing the
terminal doesn't throw the conversation away. `--continue` resumes the most
recent session **for the current project** — sessions are scoped to the
directory they ran in, so one checkout never resumes another's work.
`/sessions` lists them, `--resume <id>` picks one, and `/clear` starts a new
session without touching what is already saved.

What gets saved is the already-compacted history, so resuming costs no more
than the session did when you left it. Offloaded results are restored too, so
the `[offloaded: ...]` markers still resolve.

Files are written `0600` in a `0700` directory: a session holds whatever the
agent read, which is often source code and sometimes more. `--no-save` skips
writing entirely, and sessions older than 30 days are swept.

## Proxies

Off by default, and `HTTP_PROXY`/`ALL_PROXY` are **ignored** unless you ask
for them with `proxy = "system"` — a stray variable in your shell shouldn't
silently reroute API traffic. Settings resolve most-specific-first: `--proxy`,
then the preset's own value, then the global one.

```toml
proxy = "socks5h://127.0.0.1:1080"   # everything goes through here...

[[presets]]
name = "local-llm"
base_url = "http://localhost:11434/v1"
proxy = false                        # ...except this one, which goes direct
```

`localhost` and `127.0.0.0/8` are always reached directly, including under
`--proxy`, which otherwise overrides everything: a proxy resolves `localhost`
on its own side, so proxying a local model server would send the request to a
stranger's machine rather than yours. For a model server elsewhere on the LAN,
list it — exact host, domain suffix or CIDR:

```toml
no_proxy = ["192.168.0.0/16", ".internal", "ollama.box"]
```

`socks5h://` and `socks5://` behave identically here: httpx hands the hostname
to the proxy, so DNS is resolved on the proxy side either way.

Transient failures (429, 5xx, connect timeouts) are retried with exponential
backoff and jitter, honoring `Retry-After`. A request is only retried when
nothing has been streamed yet — never mid-answer.

## Configuration

Copy `lindwyrm.example.toml` to `./.lindwyrm.toml` (project) or
`~/.config/lindwyrm/config.toml` (user). Project settings override user ones.
The example file documents every option; the essentials:

```toml
default_preset = "flash"   # "flash", "pro", or a name from [[presets]]

# Any Anthropic- or OpenAI-compatible endpoint:
[[presets]]
name = "kimi"
format = "openai"
base_url = "https://api.tokenrouter.com/v1"
model = "moonshotai/kimi-k3-free"
api_key_env = ["TOKENROUTER_API_KEY"]
thinking = false

[policy]
read   = "allow"
write  = "confirm"
delete = "confirm"
bash   = "confirm"
bash_allowlist = ["git status", "ls"]
bash_denylist  = ["sudo", "curl"]

[[policy.rules]]        # write freely in generated/, still confirm deletes
path = "generated"
write = "allow"

[[policy.rules]]        # keep the agent out of your keys
path = "~/.ssh"
read = "deny"
```

API keys are never read from the config file itself — only from the
environment or a `key_file` you point at, so a committed config can't leak
one.

## Security notes and limits

- `bash` runs through the shell. The denylist is a backstop, not a jail — the
  real protection is that it defaults to `confirm`, so you see every command
  before it runs. For stronger isolation, run lindwyrm in a container.
- Path permissions guard against strayed file operations, and resolve
  symlinks and `../` before matching. They are not a defense against a write
  you confirm yourself.
- Summarizing is lossy by nature: it replaces older turns with a model-written
  summary, cutting only at user-turn boundaries so tool calls are never split
  from their results. Offloading, which runs first, loses nothing — the full
  text stays under `~/.local/share/lindwyrm/offload/` and is swept after 7
  days. An offloaded result is a *snapshot*, which is exactly why it is copied
  rather than re-read later: the file may have changed since.

## Tests

```bash
python -m unittest discover -s tests
```

Stdlib `unittest`, no test dependencies, about a second to run. Covers
permission resolution and path escaping, both wire formats, compaction and
offload boundaries, proxy resolution, cost accounting, session round-trips,
the streaming display, and retry backoff — including an end-to-end suite
against a fake provider served over real HTTP. CI runs it on every push
across Python 3.11, 3.12 and 3.13, and a release cannot publish on a red
suite.

## Releasing

Publishing runs on tag push via GitHub Actions using PyPI **Trusted
Publishing** (OIDC) — no API token is stored in this repo. One tag publishes
two distributions: `lindwyrm` and the `lwyrm` alias.

```bash
# bump __version__ in lindwyrm/__init__.py, commit it last, then:
git tag -a v0.5.1 -m "..." && git push origin main && git push origin v0.5.1
```

`lindwyrm/__init__.py` is the single source of truth for the version:
setuptools reads it as dynamic metadata and the workflow stamps the same
value into the alias package. Tag the commit that bumps it, and bump it last
— the workflow refuses to build when the tag doesn't match the version in the
code. Without that check the mismatch is silent, because `skip-existing`
makes uploading an already-published version succeed, so the run goes green
having published nothing.

A version on PyPI can never be re-uploaded; mistakes are fixed by releasing
forward, never by moving a tag.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 miron404.
