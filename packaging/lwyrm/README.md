# lwyrm

This is an alias package. The actual project is
**[lindwyrm](https://pypi.org/project/lindwyrm/)** — a minimal,
dependency-light CLI coding agent that talks directly to any Anthropic- or
OpenAI-compatible provider.

`lwyrm` is the name of the command, so this package exists for people who
reach for `pip install lwyrm` first. It has no code of its own; it simply
depends on `lindwyrm`, which ships the `lwyrm` command.

```bash
pip install lwyrm      # equivalent to: pip install lindwyrm
lwyrm --help
```

Source and documentation: <https://github.com/miron404/lindwyrm>
