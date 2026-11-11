"""G3: deterministic 30-case test and 40-case validation suites.

Raw CSV and private cases/metadata MUST stay out of public GitHub pending
license confirmation. All selection uses frozen data, split and predeclared
corruption rules; it never uses Gemini outputs to choose a test case.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from hajj_study.audit_data import audit
from hajj_study.corrupt import (
    DOMAINS, FIELD_DOMAIN, affected_fields, corrupt_case,
    source_evidence, stress_influence,
)
from hajj_study.proxy import risk_label, risk_score

SELECTION_SEEDS = {"test": 20261107, "validation": 20261108}
CASE_COUNTS = {
    "test": {
        "typical_clean": {"LOW": 6, "MEDIUM": 6, "HIGH": 6},
        "missing": {"LOW": 1, "MEDIUM": 1, "HIGH": 1},
        "noisy": {"LOW": 1, "MEDIUM": 1, "HIGH": 1},
        "conflicting": {"LOW": 1, "MEDIUM": 1, "HIGH": 1},
        "borderline_clean": {"LOW": 1, "MEDIUM": 1, "HIGH": 1},
    },
    "validation": {
        "typical_clean": {"LOW": 8, "MEDIUM": 8, "HIGH": 8},
        "missing": {"LOW": 2, "MEDIUM": 1, "HIGH": 1},
        "noisy": {"LOW": 1, "MEDIUM": 2, "HIGH": 1},
        "conflicting": {"LOW": 1, "MEDIUM": 1, "HIGH": 2},
        "borderline_clean": {"LOW": 2, "MEDIUM": 1, "HIGH": 1},
    },
}
GROUPS = ("typical_clean", "missing", "noisy", "conflicting", "borderline_clean")
MIN_TYPICAL_MARGIN = 0.10
MIN_BORDERLINE_MARGIN = 0.01  # avoids floating-point near-tie fixtures; P2 reviewer preference
VERSION = "cases-g3-v2"


def stable_int(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def borderline_distance(score):
    return min(abs(score - 1.8), abs(score - 2.4))


def choose_multidomain(split, kind, label, occurrence):
    # Test exactly 3 two-domain cases; validation exactly 4.
    if (kind, label) in (("missing", "HIGH"), ("conflicting", "LOW")):
        return True
    if kind == "noisy" and label == "MEDIUM":
        return True  # two such validation cases; one test case
    return False


def _read_rows(csv_path, splits_path, predictions_path):
    audit(csv_path)
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    with Path(splits_path).open(newline="", encoding="utf-8") as file:
        lookup = list(csv.DictReader(file))
    if len(lookup) != 10000 or len(rows) != 10000:
        raise ValueError("Unexpected source or split size")
    memberships = {}
    for index, item in enumerate(lookup):
        if int(item["source_row_index"]) != index or item["proxy_label"] != risk_label(rows[index]):
            raise ValueError(f"Split identity or reference label mismatch: {index}")
        memberships[index] = item["split"]
    predictions = {}
    with Path(predictions_path).open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            idx = item["source_row_index"]
            if idx in predictions or memberships[idx] != item["split"]:
                raise ValueError("Duplicate or misassigned ML prediction")
            predictions[idx] = {"predicted_risk": item["predicted_risk"],
                                "probabilities": item["probabilities"]}
    expected = {index for index in memberships if memberships[index] in ("validation", "test")}
    if set(predictions) != expected:
        raise ValueError("Missing, extra or training-set predictions in frozen ML file")
    return rows, memberships, predictions


def _assign_conflict_positions(selected):
    """Balance clean reading A/B across conflicting FIELD instances within each suite.

    This is defined before Gemini, and positions are private scorer metadata.
    """
    entries = [(item["case_id"], field) for item in selected if item["group"] == "conflicting"
               for field in item["private"]["altered_fields"]]
    entries.sort(key=lambda it: (stable_int("conflict-order", *it), it))
    positions = {}
    for i, (case, field) in enumerate(entries):
        positions[(case, field)] = (i % 2 == 0)
    if abs(sum(positions.values()) - (len(positions) - sum(positions.values()))) > 1:
        raise AssertionError("Conflict ordering unbalanced")
    return positions


def make_suite(split, rows, memberships, predictions):
    """Create a fixed suite, private reference/metadata, plus selection diagnostic."""
    rng = random.Random(SELECTION_SEEDS[split])
    counts = CASE_COUNTS[split]
    options = {label: [] for label in ("LOW", "MEDIUM", "HIGH")}
    for index, member in memberships.items():
        if member == split:
            label = risk_label(rows[index])
            options[label].append(index)
    for pool in options.values():
        pool.sort()
    used = set()
    selected = []
    attempted = Counter()
    candidates_available = {}
    # Reserve borderline rows prior to typical sampling to avoid collision.
    reserved = {}
    for label, quota in counts["borderline_clean"].items():
        pool = [idx for idx in options[label]
                if borderline_distance(risk_score(rows[idx])) >= MIN_BORDERLINE_MARGIN]
        pool.sort(key=lambda i: (borderline_distance(risk_score(rows[i])), i))
        if len(pool) < quota:
            raise ValueError(f"Not enough borderline {label} rows in {split}")
        reserved[label] = pool[:quota]
    exclusions = set(idx for values in reserved.values() for idx in values)

    def candidate_iter(group, label):
        pool = [idx for idx in options[label] if idx not in used and idx not in exclusions]
        if group == "typical_clean":
            pool = [idx for idx in pool if borderline_distance(risk_score(rows[idx])) >= MIN_TYPICAL_MARGIN]
        rng.shuffle(pool)
        candidates_available[(group, label)] = len(pool)
        return pool

    def add(group, label, idx, multi=False, metadata=None):
        if idx in used or memberships[idx] != split:
            raise AssertionError("Source rows must be unique and in the selected partition")
        used.add(idx)
        case_id = f"HJ-{'T' if split == 'test' else 'V'}{len(selected) + 1:03d}"
        selected.append({"case_id": case_id, "index": idx, "group": group, "label": label,
                         "multidomain": multi, "private": metadata or {}})

    for group in ("typical_clean", "missing", "noisy", "conflicting"):
        for label in ("LOW", "MEDIUM", "HIGH"):
            for occurrence in range(counts[group][label]):
                multi = choose_multidomain(split, group, label, occurrence)
                found = None
                for idx in candidate_iter(group, label):
                    attempted[(group, label)] += 1
                    if group == "typical_clean":
                        found = (idx, {})
                        break
                    _, private = corrupt_case(rows[idx], group, label, multi)
                    if stress_influence(rows[idx], group, private):
                        found = (idx, private)
                        break
                if found is None:
                    raise ValueError(f"No qualifying stress candidate: {split}/{group}/{label}/multi={multi}")
                add(group, label, found[0], multi, found[1])
    for label in ("LOW", "MEDIUM", "HIGH"):
        for idx in reserved[label]:
            add("borderline_clean", label, idx)

    expected = sum(sum(m.values()) for m in counts.values())
    if len(selected) != expected:
        raise AssertionError(f"Wrong number selected: {len(selected)}")
    # Opaque, deterministic case-ID assignment: IDs must not reveal class/group
    # through the original group-by-group selection order.
    random.Random(stable_int(SELECTION_SEEDS[split], "opaque-id-order")).shuffle(selected)
    for position, item in enumerate(selected, start=1):
        item["case_id"] = f"HJ-{'T' if split == 'test' else 'V'}{position:03d}"
    conflict_positions = _assign_conflict_positions(selected)
    public_cases, gold_records, clean_refs, private_records = [], [], [], []
    for item in selected:
        idx, group, label, case_id = item["index"], item["group"], item["label"], item["case_id"]
        clean = source_evidence(rows[idx])
        if group in ("missing", "noisy", "conflicting"):
            fields = affected_fields(group, label, item["multidomain"])
            positions = {field: conflict_positions[(case_id, field)] for field in fields} if group == "conflicting" else {}
            observed, metadata = corrupt_case(rows[idx], group, label, item["multidomain"], positions)
            if not stress_influence(rows[idx], group, metadata):
                raise AssertionError("Previously qualifying corruption became noninfluential")
        else:
            observed, metadata = clean, {"kind": None, "altered_fields": [],
                                          "altered_domains": [], "clean_first_by_field": {},
                                          "alternative_values_for_private_analysis": {}}
        ml = predictions[idx]
        public_cases.append({"case_id": case_id, "observations": observed,
                             "ml_prediction": ml})
        gold_records.append({"case_id": case_id, "risk_level": label})
        if group in ("missing", "noisy", "conflicting"):
            clean_refs.append({"case_id": case_id, "observations": clean,
                               "ml_prediction": ml})
        private_records.append({
            "case_id": case_id, "source_row_index_zero_based": idx,
            "case_group": group, "gold": label, "clean_score": risk_score(rows[idx]),
            "source_clean_hash": digest(clean), "observed_hash": digest(observed),
            "corruption_metadata": metadata,
            "stress_conditioned_selection": group in ("missing", "noisy", "conflicting"),
            "borderline_distance": borderline_distance(risk_score(rows[idx])),
        })
    expected_refs = 9 if split == "test" else 12
    if len(clean_refs) != expected_refs:
        raise AssertionError("Incorrect clean reference count")
    multi_cases = sum(len(set(x["corruption_metadata"]["altered_domains"])) > 1 for x in private_records)
    if multi_cases != (3 if split == "test" else 4):
        raise AssertionError(f"Expected 3/4 multi-domain cases, got {multi_cases}")
    return {"public_cases": public_cases, "gold": gold_records,
            "clean_references": clean_refs, "private_metadata": private_records,
            "diagnostics": {"source_partition": split, "class_counts": dict(Counter(x["gold"] for x in private_records)),
                            "group_counts": dict(Counter(x["case_group"] for x in private_records)),
                            "multi_domain_corruptions": multi_cases,
                            "candidate_trials": {f"{a}/{b}": n for (a,b),n in sorted(attempted.items())},
                            "stress_selection_warning": "Corrupted cases are selected for diagnostic influence, not random prevalence"}}


def _write_jsonl(path, items):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_suite(csv_path, splits_path, predictions_path, out_dir):
    rows, memberships, predictions = _read_rows(csv_path, splits_path, predictions_path)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"generation_version": VERSION,
                "input_hashes": {"raw_csv": sha(csv_path), "row_splits": sha(splits_path),
                                 "shared_predictions": sha(predictions_path)},
                "selection_seeds": SELECTION_SEEDS,
                "case_id_policy": "seeded shuffled sequence; not ordered by class/group",
                "code_hashes": {
                    "proxy_py": sha(Path(__file__).with_name("proxy.py")),
                    "corrupt_py": sha(Path(__file__).with_name("corrupt.py")),
                    "cases_py": sha(Path(__file__)),
                },
                "label_policy": "gold from clean source; intentional class balancing and stress conditioning",
                "suite_files": {}}
    for split in ("test", "validation"):
        suite = make_suite(split, rows, memberships, predictions)
        target = dest / split
        target.mkdir(parents=True, exist_ok=True)
        filenames = {
            "cases": ("cases.jsonl", "public_cases"),
            "gold_labels": ("gold_labels.jsonl", "gold"),
            "clean_references": ("clean_references.jsonl", "clean_references"),
            "private_metadata": ("private_metadata.jsonl", "private_metadata"),
        }
        hashes = {}
        for role, (filename, key) in filenames.items():
            path = target / filename
            _write_jsonl(path, suite[key])
            hashes[role] = {"file": f"{split}/{filename}", "sha256": sha(path), "n": len(suite[key])}
        (target / "diagnostics.json").write_text(json.dumps(suite["diagnostics"], indent=2) + "\n")
        manifest["suite_files"][split] = hashes
        manifest["suite_files"][split]["diagnostics"] = suite["diagnostics"]
    (dest / "case_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    print(json.dumps(build_suite(args.csv, args.splits, args.predictions, args.out), indent=2))


if __name__ == "__main__":
    main()
