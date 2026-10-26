"""Check window selection and focus boundaries without desktop input."""
from dataclasses import replace
import unittest

from vimm8bdm.desktop import Focus
from vimm8bdm.windows import WindowInfo, candidates, focus_matches


class FakeWindows:
    def __init__(self):
        self.info = WindowInfo(100, 42, "Zandronum", "Private title", 640, 480, True)
        self.focus = 100
        self.parents = {101: 100, 102: 101, 200: 300, 300: 0}
        self.closed = False

    def describe(self, unused):
        return self.info

    def current(self):
        return self.focus

    def parent_of(self, window):
        return self.parents.get(window)

    def close(self):
        self.closed = True


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.windows = FakeWindows()
        self.focus = Focus(100, self.windows)

    def test_target_and_descendant_focus_match(self):
        self.assertTrue(self.focus())
        self.windows.focus = 102
        self.assertTrue(self.focus())

    def test_ancestor_and_unrelated_focus_do_not_match(self):
        for selected in (0, 1, 200, 300):
            self.windows.focus = selected
            self.assertFalse(self.focus())

    def test_parent_focus_is_not_child_focus(self):
        self.assertFalse(focus_matches(101, 100, self.windows.parent_of))

    def test_cyclic_parent_chain_fails_closed(self):
        self.assertFalse(focus_matches(100, 200, lambda w: 300 if w == 200 else 200))

    def test_destroyed_window_fails_closed(self):
        self.windows.info = None
        self.assertFalse(self.focus())

    def test_reused_window_with_another_process_fails_closed(self):
        self.windows.info = replace(self.windows.info, process_id=43)
        self.assertFalse(self.focus())

    def test_changed_class_fails_closed(self):
        self.windows.info = replace(self.windows.info, class_name="Other")
        self.assertFalse(self.focus())

    def test_hidden_or_small_window_blocks_input(self):
        original = self.windows.info
        for change in ({"visible": False}, {"width": 10}, {"height": 10}):
            self.windows.info = replace(original, **change)
            self.assertFalse(self.focus())

    def test_title_changes_do_not_change_identity(self):
        self.windows.info = replace(self.windows.info, title="A changed game title")
        self.assertTrue(self.focus())

    def test_metadata_errors_fail_closed(self):
        def unavailable():
            raise OSError("Display connection lost")
        self.windows.current = unavailable
        self.assertFalse(self.focus())

    def test_diagnostic_view_excludes_titles_by_default(self):
        self.assertNotIn("title", self.windows.info.diagnostic_view())
        self.assertIn("title", self.windows.info.diagnostic_view(True))

    def test_candidates_exclude_startup_dialog_and_other_apps(self):
        game = self.windows.info
        startup = replace(game, window_id=400, width=10, height=10)
        same_process = replace(game, window_id=401, class_name="", title="")
        other = replace(game, window_id=500, process_id=99, class_name="Other", title="Other")
        self.assertEqual(candidates([startup, same_process, other]), [same_process])
        self.assertEqual(candidates([startup, same_process, other], True), [same_process, other])

    def test_invalid_target_closes_display(self):
        self.windows.info = None
        with self.assertRaises(ValueError):
            Focus(400, self.windows)
        self.assertTrue(self.windows.closed)


if __name__ == "__main__":
    unittest.main()
