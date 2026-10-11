"""Regression coverage for the mini-CTK cache save boundary."""

from pathlib import Path
import re
import unittest


ACTION = Path(__file__).parents[3] / ".github/actions/fetch_ctk/action.yml"


def should_save(ref: str, default_branch: str, cancelled: bool, cache_hit: str) -> bool:
    """Model the save condition using GitHub's full ref and cache-hit output."""
    return (
        not cancelled
        and ref == f"refs/heads/{default_branch}"
        and cache_hit != "true"
    )


class TestCtkCacheSaveGate(unittest.TestCase):
    def test_workflow_condition_uses_full_default_branch_ref_and_preserves_guards(self):
        action = ACTION.read_text()
        match = re.search(
            r"(?ms)^    - name: Upload CTK cache\n      if: \$\{\{(.*?)\}\}",
            action,
        )
        self.assertIsNotNone(match)
        expression = " ".join(match.group(1).split())
        self.assertIn("!cancelled()", expression)
        self.assertIn(
            "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)",
            expression,
        )
        self.assertIn("steps.ctk-get-cache.outputs.cache-hit != 'true'", expression)
        self.assertIn(
            "- name: Get CUDA components\n      if: ${{ steps.ctk-get-cache.outputs.cache-hit != 'true' }}",
            action,
        )
        self.assertIn(
            "- name: Restore CTK cache\n      if: ${{ steps.ctk-get-cache.outputs.cache-hit == 'true' }}",
            action,
        )

    def test_cache_save_ref_and_status_matrix(self):
        cases = (
            # The default branch saves after a miss, including an empty output.
            ("refs/heads/main", "main", False, "false", True),
            ("refs/heads/main", "main", False, "", True),
            # Other branches and PR merge refs restore but never save.
            ("refs/heads/feature", "main", False, "false", False),
            ("refs/pull/3071/merge", "main", False, "false", False),
            # Full-ref comparison prevents a same-named tag from saving.
            ("refs/tags/main", "main", False, "false", False),
            # The repository's configured default branch may differ from main.
            ("refs/heads/ctk-next", "ctk-next", False, "false", True),
            ("refs/heads/main", "main", True, "false", False),
            ("refs/heads/main", "main", False, "true", False),
        )
        for ref, default_branch, cancelled, cache_hit, expected in cases:
            with self.subTest(ref=ref, cancelled=cancelled, cache_hit=cache_hit):
                self.assertEqual(
                    should_save(ref, default_branch, cancelled, cache_hit), expected
                )


if __name__ == "__main__":
    unittest.main()
