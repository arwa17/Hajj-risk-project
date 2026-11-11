"""Checks public synthetic end-to-end demo; no private data or API calls."""
import unittest

from examples.offline_demo import run_demo


class OfflineDemoTests(unittest.TestCase):
    def test_three_real_pipeline_paths_with_scripted_provider(self):
        rows = run_demo()
        self.assertEqual(len(rows), 3)
        self.assertEqual([r['single_status'] for r in rows], ['ISSUE', 'ISSUE', 'ABSTAIN'])
        self.assertEqual([r['multi_status'] for r in rows], ['ISSUE', 'ISSUE', 'ABSTAIN'])
        self.assertEqual([r['multi_scripted_calls'] for r in rows], [4, 6, 6])


if __name__ == '__main__':
    unittest.main()
