"""Read-only provenance, schema, and reconstructed-label audit for the raw CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from hajj_study.proxy import PROXY_FIELDS, risk_label

EXPECTED_SHA256 = "7c289edf5b3390b479c48ac5a493db566f6c46d4d587ae101b03d885bf948e8a"
EXPECTED_ROWS = 10000
EXPECTED_COLUMNS = 30
EXPECTED_LABEL_COUNTS = {"LOW": 3337, "MEDIUM": 5755, "HIGH": 908}


def audit(csv_path: str | Path) -> dict:
    path = Path(csv_path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA256:
        raise ValueError(f"Raw CSV SHA mismatch: {sha}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != EXPECTED_COLUMNS or len(set(fields)) != len(fields):
            raise ValueError(f"Expected {EXPECTED_COLUMNS} unique columns")
        missing_fields = set(PROXY_FIELDS) - set(fields)
        if missing_fields:
            raise ValueError(f"Missing proxy fields: {sorted(missing_fields)}")
        if "Risk_Label" in fields or "Operational_Risk_Score" in fields:
            raise ValueError("Unexpected source label column; recheck source provenance")
        seen = set()
        labels = Counter()
        missing_cells = 0
        rows = 0
        for rows, row in enumerate(reader, start=1):
            if None in row:
                raise ValueError(f"Extra cells detected on CSV row {rows}")
            signature = tuple(row[field] for field in fields)
            if signature in seen:
                raise ValueError(f"Duplicate row at position {rows}")
            seen.add(signature)
            if any(value is None or value == "" for value in signature):
                missing_cells += sum(value is None or value == "" for value in signature)
            labels[risk_label(row)] += 1
        if rows != EXPECTED_ROWS or missing_cells:
            raise ValueError(f"Expected {EXPECTED_ROWS} complete rows; got {rows}, missing cells {missing_cells}")
        if dict(labels) != EXPECTED_LABEL_COUNTS:
            raise ValueError(f"Proxy label distribution differs: {dict(labels)}")
    return {
        "csv_sha256": sha,
        "rows": rows,
        "columns": len(fields),
        "null_or_blank_cells": missing_cells,
        "duplicate_rows": 0,
        "proxy_labels": {key: labels[key] for key in ("LOW", "MEDIUM", "HIGH")},
        "published_reference_distribution": {"LOW": 3121, "MEDIUM": 5869, "HIGH": 1010},
        "encoding": "reconstructed; not exact publication replication",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--report", help="Optional audit JSON output path")
    args = parser.parse_args()
    report = audit(args.csv)
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
