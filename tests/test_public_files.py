"""Check index content even when the working copy contains different text."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("public_check", Path(__file__).parents[1] / "scripts/check_public_files.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class PublicFileTests(unittest.TestCase):
    def check_index(self, staged, working):
        with tempfile.TemporaryDirectory(prefix="vimm8bdm-index-") as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-b", "main", directory], check=True, capture_output=True)
            (root / "README.md").write_text(staged)
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True, capture_output=True)
            (root / "README.md").write_text(working)
            with patch.object(CHECK, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                return CHECK.main()

    def test_staged_private_path_is_not_hidden_by_working_copy(self):
        machine_path = "/" + "home" + "/" + "example" + "/project"
        self.assertEqual(self.check_index(machine_path, "Public instructions."), 1)

    def test_unstaged_local_path_does_not_change_public_index(self):
        machine_path = "/" + "home" + "/" + "example" + "/project"
        self.assertEqual(self.check_index("Public instructions.", machine_path), 0)


if __name__ == "__main__":
    unittest.main()
