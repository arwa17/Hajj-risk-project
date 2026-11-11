"""Offline tests for private study logging; fixtures are not Gemini evidence."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from hajj_study.study_log import sync_index, record_activity, mirror_summary, run_g5


class StudyLogTests(unittest.TestCase):
    def test_recoverable_per_case_index_without_raw_response(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / "private"
            src.mkdir()
            record = {"case_id": "HJ-V001", "probe_id": "smoke-01", "seed": 42,
                      "pre_sha256": "a" * 64, "accepted": True,
                      "call": {"model_version": "mock-model", "report": {"case_id": "HJ-V001"},
                               "raw_response_private": "PRIVATE RAW RESPONSE SHOULD NOT COPY",
                               "elapsed_ms": 12, "attempts": [{"attempt": 1, "status": "STOP",
                                 "finish_reason": "STOP", "usage": {"prompt_tokens": 12,
                                 "candidates_tokens": 3, "thoughts_tokens": 4}}]}}
            (src / "smoke.jsonl").write_text(json.dumps(record) + "\n")
            log = root / "logs"
            self.assertEqual(sync_index(src, log), (1, 1))
            self.assertEqual(sync_index(src, log), (0, 1))
            details = list((log / "cases" / "HJ-V001").glob("*.json"))
            self.assertEqual(len(details), 1)
            case = json.loads(details[0].read_text())
            self.assertEqual(case["prompt_tokens"], 12)
            self.assertEqual(case["thoughts_tokens"], 4)
            self.assertEqual(case["candidates_tokens"], 3)
            self.assertEqual(case["status"], "ACCEPTED")
            self.assertNotIn("PRIVATE RAW RESPONSE", details[0].read_text())
            self.assertIn("Record HJ-V001", (log / "PROGRESS.md").read_text())

    def test_failed_validation_logged_as_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            src = p / "private"
            src.mkdir()
            (src / "validation_runs.jsonl").write_text(json.dumps({"case_id": "HJ-V002",
                "seed": 42, "status": "FAILED", "failure": {"category": "RATE_LIMIT"},
                "verification_count": 0}) + "\n")
            sync_index(src, p / "logs")
            row = json.loads((p / "logs" / "case_index.jsonl").read_text())
            self.assertEqual(row["status"], "FAILED")
            self.assertEqual(row["failure_category"], "RATE_LIMIT")

    def test_failure_records_provider_attempt_usage(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "validation_runs.jsonl").write_text(json.dumps({
                "case_id": "HJ-V020", "status": "FAILED", "seed": 42,
                "failure": {"category": "RATE_LIMIT", "attempts": [
                    {"attempt": 1, "status": "HTTP_429", "usage": {"prompt_tokens": 10,
                        "candidates_tokens": 0, "thoughts_tokens": 0}}]}}) + "\n")
            sync_index(p, p / "logs")
            record = json.loads((p / "logs/case_index.jsonl").read_text())
            self.assertEqual(record["attempts"], 1)
            self.assertEqual(record["prompt_tokens"], 10)
            self.assertEqual(record["attempt_details"][0]["phase"], "failed_call")

    def test_partial_line_not_counted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p / "smoke.jsonl").write_text('{"case_id": "incomplete"}')
            self.assertEqual(sync_index(p, p / "logs"), (0,0))

    def test_mirror_excludes_private_g5_raw(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            log=p / "logs"
            log.mkdir()
            record_activity(log, "G5", "test", "PASS", "Safe information only")
            cloud=p / "Drive"
            cloud.mkdir()
            self.assertTrue(mirror_summary(log, cloud))
            self.assertTrue((cloud / "PROGRESS.md").exists())
            self.assertFalse((cloud / "smoke.jsonl").exists())

    def test_g5_run_does_not_sync_to_cloud_during_execution(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            private = p / "g5"
            private.mkdir()
            (private / "manifest_pre.json").write_text("{}")
            cloud = p / "cloud"
            cloud.mkdir()
            with patch("hajj_study.study_log.subprocess.Popen") as mocked:
                child = mocked.return_value
                child.poll.return_value = 0
                child.wait.return_value = 0
                code = run_g5("smoke", private, p / "logs", cloud)
            self.assertEqual(code, 0)
            self.assertFalse(list(cloud.iterdir()))
            self.assertTrue((p / "logs" / "PROGRESS.md").exists())

    def test_smoke_command_rejects_missing_preflight_before_invocation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            with self.assertRaisesRegex(ValueError, "missing manifest_pre"):
                run_g5("smoke", p / "g5", p / "logs", None)


if __name__ == "__main__":
    unittest.main()
