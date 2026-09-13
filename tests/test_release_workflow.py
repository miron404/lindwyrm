"""The alias package's version is stamped by CI, and the stamping broke.

It broke in the worst possible way: the regex matched nothing, the alias was
built with the previous version, `skip-existing` skipped the upload, and the
run went green having published nothing. Nobody noticed for a release because
every version bump until then had also hand-edited the alias file.

So the workflow's own script is pulled out of the YAML and run against the
real alias file here, where it costs a second instead of a version number.
"""

import pathlib
import re
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
ALIAS = ROOT / "packaging" / "lwyrm" / "pyproject.toml"


def workflow_step_script(step_name: str) -> str:
    """The python heredoc belonging to one named workflow step."""
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index(f"- name: {step_name}")
    body = text[start:]
    open_at = body.index("<<'PY'\n") + len("<<'PY'\n")
    close_at = body.index("\n          PY\n", open_at)
    return textwrap.dedent(body[open_at:close_at])


class TestAliasVersionStamping(unittest.TestCase):
    def setUp(self):
        self.script = workflow_step_script("Sync alias version with lindwyrm")

    def run_script(self, tmp: pathlib.Path, version: str) -> str:
        (tmp / "lindwyrm").mkdir(parents=True, exist_ok=True)
        (tmp / "lindwyrm" / "__init__.py").write_text(
            f'__version__ = "{version}"\n', encoding="utf-8")
        target = tmp / "packaging" / "lwyrm" / "pyproject.toml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(ALIAS.read_text(encoding="utf-8"), encoding="utf-8")

        import contextlib
        import io
        import os
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(self.script, "publish.yml", "exec"),
                     {"__name__": "__main__"})
        finally:
            os.chdir(cwd)
        return target.read_text(encoding="utf-8")

    def test_it_stamps_the_real_alias_file(self):
        """Against the file as it actually is, comment and all."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = self.run_script(pathlib.Path(d), "9.9.9")
        self.assertRegex(out, r'(?m)^version = "9\.9\.9"')
        self.assertIn('lindwyrm>=9.9.9', out)

    def test_the_comment_on_the_version_line_survives(self):
        """It explains why the value is not maintained by hand; losing it is
        how someone edits it by hand again."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = self.run_script(pathlib.Path(d), "9.9.9")
        line = next(l for l in out.splitlines() if l.startswith("version = "))
        self.assertIn("#", line, "the explanatory comment was stamped away")


class TestAliasFileShape(unittest.TestCase):
    def test_version_and_dependency_are_on_their_own_lines(self):
        text = ALIAS.read_text(encoding="utf-8")
        self.assertRegex(text, r'(?m)^version = "[0-9]')
        self.assertRegex(text, r'lindwyrm>=[0-9]')

    def test_skip_existing_is_paired_with_a_stamping_check(self):
        """skip-existing turns a bad build into a green run that publishes
        nothing, so the stamp must be verified before anything is built."""
        wf = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("skip-existing: true", wf)
        self.assertIn("Failed to stamp the alias package", wf)


if __name__ == "__main__":
    unittest.main()
