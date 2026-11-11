"""Frozen stratified source-row split: 7,000 train / 1,000 validation / 2,000 test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from hajj_study.audit_data import audit
from hajj_study.feature_contract import validate_features
from hajj_study.proxy import risk_label

SEED = 42
EXPECTED_COUNTS = {"train": 7000, "validation": 1000, "test": 2000}


def split_indices(labels: list[str], seed: int = SEED) -> dict[int, str]:
    indices = list(range(len(labels)))
    dev, test = train_test_split(indices, test_size=0.2, random_state=seed, stratify=labels)
    train, val = train_test_split(
        dev, test_size=0.125, random_state=seed,
        stratify=[labels[i] for i in dev],
    )
    result = {int(i): "train" for i in train}
    result.update({int(i): "validation" for i in val})
    result.update({int(i): "test" for i in test})
    if len(result) != len(indices) or sorted(result) != indices:
        raise AssertionError("Train/validation/test source row indices overlap or omit records")
    return result


def create_split(source_csv: str | Path, output_csv: str | Path) -> dict:
    audit(source_csv)
    with Path(source_csv).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        validate_features(reader.fieldnames or [])
        labels = [risk_label(row) for row in reader]
    groups = split_indices(labels)
    counts = Counter(groups.values())
    if dict(counts) != EXPECTED_COUNTS:
        raise AssertionError(f"Unexpected split sizes: {dict(counts)}")
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["source_row_index", "split", "proxy_label"])
        for row_index in range(len(labels)):
            writer.writerow([row_index, groups[row_index], labels[row_index]])
    by_label = {
        group: {cls: sum(1 for i, g in groups.items() if g == group and labels[i] == cls)
                for cls in ("LOW", "MEDIUM", "HIGH")}
        for group in ("train", "validation", "test")
    }
    return {
        "random_state": SEED,
        "counts": {g: counts[g] for g in ("train", "validation", "test")},
        "label_counts": by_label,
        "row_splits_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    report = create_split(args.csv, args.out)
    txt = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        Path(args.report).write_text(txt, encoding="utf-8")
    print(txt, end="")


if __name__ == "__main__":
    main()
