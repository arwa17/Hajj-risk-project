"""G3 acceptance tests for frozen data suites (requires local input artifacts).

STUDY_RAW_CSV, STUDY_SPLITS, STUDY_PREDICTIONS must point to private files for
integration tests; public CI does not bundle licensed data.
"""
import csv
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from hajj_study.cases import build_suite, sha
from hajj_study.corrupt import affected_fields, stress_influence, source_evidence
from hajj_study.proxy import risk_label


def parse(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


class IntegrationSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = os.getenv("STUDY_RAW_CSV")
        cls.splits = os.getenv("STUDY_SPLITS")
        cls.predictions = os.getenv("STUDY_PREDICTIONS")
        if not (cls.csv_path and cls.splits and cls.predictions):
            raise unittest.SkipTest("Private CSV/splits/model predictions are not provided")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.tmp2 = tempfile.TemporaryDirectory()
        cls.manifest = build_suite(cls.csv_path, cls.splits, cls.predictions, cls.tmp.name)
        cls.manifest2 = build_suite(cls.csv_path, cls.splits, cls.predictions, cls.tmp2.name)
        with open(cls.csv_path, newline="", encoding="utf-8-sig") as f:
            cls.raw = list(csv.DictReader(f))

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()
        if hasattr(cls, "tmp2"):
            cls.tmp2.cleanup()

    def records(self, split, file):
        return parse(Path(self.tmp.name) / split / file)

    def test_frozen_deterministic_all_artifacts(self):
        for split in ("test", "validation"):
            for filename in ("cases.jsonl", "gold_labels.jsonl", "clean_references.jsonl",
                             "private_metadata.jsonl", "diagnostics.json"):
                a = Path(self.tmp.name) / split / filename
                b = Path(self.tmp2.name) / split / filename
                self.assertEqual(sha(a), sha(b), (split, filename))
        self.assertEqual(self.manifest, self.manifest2)
        self.assertEqual(sha(Path(self.tmp.name) / "case_manifest.json"),
                         sha(Path(self.tmp2.name) / "case_manifest.json"))

    def test_exact_counts_class_balance_and_no_split_overlap(self):
        ids = {}
        for split, expected_n, expected_count, refs, multi in (
            ("test", 30, {"LOW": 10, "MEDIUM": 10, "HIGH": 10}, 9, 3),
            ("validation", 40, {"LOW": 14, "MEDIUM": 13, "HIGH": 13}, 12, 4),
        ):
            meta = self.records(split, "private_metadata.jsonl")
            gold = self.records(split, "gold_labels.jsonl")
            public = self.records(split, "cases.jsonl")
            clean = self.records(split, "clean_references.jsonl")
            self.assertEqual(len(meta), expected_n)
            self.assertEqual(len(gold), expected_n)
            self.assertEqual(len(public), expected_n)
            self.assertEqual(len(clean), refs)
            self.assertEqual(Counter(x["risk_level"] for x in gold), expected_count)
            self.assertEqual(len({x["case_id"] for x in meta}), expected_n)
            self.assertEqual(Counter(x["case_group"] for x in meta),
                             {"typical_clean": 18 if split == "test" else 24,
                              "missing": 3 if split == "test" else 4,
                              "noisy": 3 if split == "test" else 4,
                              "conflicting": 3 if split == "test" else 4,
                              "borderline_clean": 3 if split == "test" else 4})
            self.assertEqual(sum(len(x["corruption_metadata"]["altered_domains"]) == 2 for x in meta), multi)
            ids[split] = {x["source_row_index_zero_based"] for x in meta}
            self.assertEqual(len(ids[split]), expected_n)
        self.assertFalse(ids["test"] & ids["validation"])
        with open(self.splits, newline="", encoding="utf-8") as f:
            membership = {int(x["source_row_index"]): x["split"] for x in csv.DictReader(f)}
        for split in ids:
            self.assertTrue(all(membership[i] == split for i in ids[split]))

    def test_case_payload_no_label_corruption_or_source_row_leak(self):
        for split in ("test", "validation"):
            for public in self.records(split, "cases.jsonl"):
                self.assertEqual(set(public), {"case_id", "observations", "ml_prediction"})
                self.assertEqual(len(public["observations"]), 7)
                self.assertEqual(set(public["ml_prediction"]), {"predicted_risk", "probabilities"})
                for field in ("gold", "risk_label", "source_row_index", "case_group", "kind",
                              "altered_fields", "clean", "clean_first_by_field", "label_score"):
                    self.assertNotIn(field, json.dumps(public).lower())

    def test_reference_gold_is_clean_source_only_and_stress_is_influential(self):
        for split in ("test", "validation"):
            public = {x["case_id"]: x for x in self.records(split, "cases.jsonl")}
            meta = {x["case_id"]: x for x in self.records(split, "private_metadata.jsonl")}
            gold = {x["case_id"]: x["risk_level"] for x in self.records(split, "gold_labels.jsonl")}
            refs = {x["case_id"]: x for x in self.records(split, "clean_references.jsonl")}
            for case_id, m in meta.items():
                clean = self.raw[m["source_row_index_zero_based"]]
                self.assertEqual(risk_label(clean), gold[case_id])
                self.assertEqual(m["gold"], gold[case_id])
                self.assertEqual(public[case_id]["ml_prediction"], refs.get(case_id, public[case_id])["ml_prediction"])
                if m["case_group"] in ("missing", "noisy", "conflicting"):
                    self.assertTrue(stress_influence(clean, m["case_group"], m["corruption_metadata"]))
                    self.assertEqual(refs[case_id]["observations"], source_evidence(clean))
                    self.assertNotEqual(public[case_id]["observations"], refs[case_id]["observations"])
                else:
                    self.assertEqual(public[case_id]["observations"], source_evidence(clean))

    def test_conflict_order_balanced_and_no_type_sidechannel(self):
        for split in ("test", "validation"):
            public = {x["case_id"]: x for x in self.records(split, "cases.jsonl")}
            metadata = self.records(split, "private_metadata.jsonl")
            positions = []
            for m in metadata:
                if m["case_group"] != "conflicting":
                    continue
                case_id = m["case_id"]
                clean = source_evidence(self.raw[m["source_row_index_zero_based"]])
                for field in m["corruption_metadata"]["altered_fields"]:
                    pair = public[case_id]["observations"][field]
                    self.assertEqual(type(pair[0]["value"]), type(pair[1]["value"]))
                    self.assertEqual([x["source"] for x in pair], ["reading_a", "reading_b"])
                    p = pair[0]["value"] == clean[field]
                    self.assertNotEqual(pair[1]["value"] == clean[field], p)
                    self.assertEqual(m["corruption_metadata"]["clean_first_by_field"][field], p)
                    positions.append(p)
            self.assertLessEqual(abs(positions.count(True) - positions.count(False)), 1)

    def test_hash_manifest_matches_files(self):
        for split, items in self.manifest["suite_files"].items():
            for role, value in items.items():
                if role == "diagnostics":
                    continue
                path = Path(self.tmp.name) / value["file"]
                self.assertEqual(value["sha256"], sha(path))


if __name__ == "__main__":
    unittest.main()
