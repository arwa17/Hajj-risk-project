"""Private append-only execution ledger and per-case index for CPCS 683.

Runs the existing, unchanged G5 CLI while indexing each completed JSONL record.
This is not a Gemini client and does not validate whether a response came from Gemini.
Raw model outputs remain in the original private G5 JSONL files; never sync gold.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

STAGES = {"smoke": ("smoke.jsonl", "G5_SMOKE"),
          "validation": ("validation_runs.jsonl", "G5_VALIDATION")}
CSV_FIELDS = ["recorded_utc", "stage", "case_id", "probe_id", "seed",
              "condition", "role", "status", "model_version", "attempts",
              "prompt_tokens", "candidates_tokens", "thoughts_tokens", "elapsed_ms",
              "verification_domain", "verification_count", "failure_category", "source_sha256"]


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, entry: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _iter_lines(path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n"):
                # Writer is still appending. Do not index a partial record.
                break
            if line.strip():
                yield line


def _safe_case_id(value):
    raw = str(value or "unknown")
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:100]


def _failure_kind(record):
    problem = record.get("failure")
    if isinstance(problem, dict):
        return str(problem.get("category", "PROVIDER_FAILURE"))[:120]
    return "GATE_OR_SCHEMA_FAILURE" if problem else ""


def _attempt_records(record):
    if "call" in record and isinstance(record["call"], dict):
        return [("smoke", record["call"])]
    calls = [(phase, record[phase]) for phase in ("initial", "post")
             if isinstance(record.get(phase), dict)]
    if isinstance(record.get("failure"), dict):
        failed_attempts = record["failure"].get("attempts") or []
        if failed_attempts:
            calls.append(("failed_call", {"attempts": failed_attempts}))
    return calls


def _summarize(record: dict, source_name: str, hash_: str):
    stage = "G5_SMOKE" if source_name == "smoke.jsonl" else "G5_VALIDATION"
    calls = _attempt_records(record)
    attempts = []
    reports = {}
    models = []
    elapsed = 0
    totals = {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0}
    for phase, call in calls:
        if call.get("report") is not None:
            reports[phase] = call["report"]  # model inference only; never a gold label
        if call.get("model_version"):
            models.append(str(call["model_version"]))
        elapsed += float(call.get("elapsed_ms") or 0)
        for attempt in call.get("attempts", []):
            usage = attempt.get("usage") or {}
            safe_attempt = {
                "phase": phase,
                "attempt": attempt.get("attempt"),
                "status": attempt.get("status"),
                "finish_reason": attempt.get("finish_reason"),
                "elapsed_ms": attempt.get("elapsed_ms"),
                "response_sha256": attempt.get("response_sha256"),
                "model_version": attempt.get("model_version"),
                "usage": {key: usage.get(key) for key in totals},
            }
            attempts.append(safe_attempt)
            for key in totals:
                value = usage.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[key] += value
    accepted = record.get("accepted") is True if stage == "G5_SMOKE" else record.get("status") == "OK"
    return {
        "recorded_utc": _utc(), "stage": stage,
        "case_id": str(record.get("case_id", "")),
        "probe_id": record.get("probe_id"), "seed": record.get("seed"),
        "condition": "Single-Agent", "role": "Generalist",
        "status": "ACCEPTED" if accepted else "FAILED",
        "source_filename": source_name, "source_sha256": hash_,
        "pre_sha256": record.get("pre_sha256"),
        "model_version": models[0] if models and len(set(models)) == 1 else None,
        "attempts": len(attempts), "attempt_details": attempts,
        **totals, "elapsed_ms": round(elapsed, 2),
        "verification_domain": record.get("verification_domain"),
        "verification_count": record.get("verification_count", 0),
        "parsed_inference_reports": reports,
        "failure_category": _failure_kind(record),
        "gate_outcome": "NOT_FROZEN_AT_G5_VALIDATION",
        "raw_trace_location": source_name,  # canonical raw is intentionally not copied
        "evidence_class": "RUNTIME_RECORDED__NOT_INDEPENDENT_PROVIDER_VERIFICATION",
    }


def _read_index(path: Path):
    entries = []
    for raw in _iter_lines(path):
        entries.append(json.loads(raw))
    return entries


def _write_csv(log_dir: Path, rows: list[dict]):
    # Atomic to avoid partial CSV if a run is interrupted.
    path = log_dir / "case_summary.csv"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows({k: row.get(k, "") for k in CSV_FIELDS} for row in rows)
    os.replace(tmp, path)


def record_activity(log_dir: Path, stage: str, action: str, outcome: str, details: str):
    entry = {"timestamp_utc": _utc(), "stage": stage, "action": action,
             "outcome": outcome, "details": details}
    _append_jsonl(log_dir / "activity.jsonl", entry)
    with (log_dir / "PROGRESS.md").open("a", encoding="utf-8") as f:
        f.write(f"\n### {entry['timestamp_utc']} — {stage}: {action}\n"
                f"Result: **{outcome}**\n\n{details}\n")
        f.flush()
        os.fsync(f.fileno())
    return entry


def sync_index(private_output: Path, log_dir: Path):
    """Recoverable, idempotent indexing; no source file is modified."""
    log_dir.mkdir(parents=True, exist_ok=True)
    existing = _read_index(log_dir / "case_index.jsonl")
    known = {row["source_sha256"] for row in existing}
    added = 0
    for source_name, stage in STAGES.values():
        for raw in _iter_lines(private_output / source_name):
            hash_ = hashlib.sha256((source_name + "\0" + raw).encode()).hexdigest()
            if hash_ in known:
                continue
            source = json.loads(raw)
            if not isinstance(source, dict):
                raise ValueError(f"Invalid private run record in {source_name}")
            row = _summarize(source, source_name, hash_)
            cid = _safe_case_id(row["case_id"])
            label = _safe_case_id(row["probe_id"] or "validation")
            case_path = log_dir / "cases" / cid / f"{stage}_{label}_{row['seed']}_{hash_[:12]}.json"
            _atomic_write(case_path, json.dumps(row, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
            _append_jsonl(log_dir / "case_index.jsonl", row)
            existing.append(row)
            known.add(hash_)
            added += 1
            record_activity(log_dir, stage, f"Record {cid} {label}", row["status"],
                            f"Seed={row['seed']}; attempts={row['attempts']}; "
                            f"verification_count={row['verification_count']}; "
                            f"model={row['model_version'] or 'not verified'}; "
                            f"details saved privately: {case_path.relative_to(log_dir)}")
    _write_csv(log_dir, existing)
    return added, len(existing)


def mirror_summary(log_dir: Path, cloud_dir: Path | None):
    if cloud_dir is None:
        return False
    cloud_dir = cloud_dir.expanduser().resolve()
    if not cloud_dir.is_dir():
        raise ValueError(f"Cloud sync directory does not exist: {cloud_dir}")
    if not os.access(cloud_dir, os.W_OK):
        raise ValueError("Cloud sync directory is not writable")
    # Never include canonical raw output, gold labels or API keys in mirror.
    for name in ("PROGRESS.md", "activity.jsonl", "case_index.jsonl", "case_summary.csv"):
        src = log_dir / name
        if src.exists():
            shutil.copy2(src, cloud_dir / name)
    src_cases = log_dir / "cases"
    if src_cases.is_dir():
        shutil.copytree(src_cases, cloud_dir / "cases", dirs_exist_ok=True)
    return True


def run_g5(step: str, private_output: Path, log_dir: Path, cloud_dir: Path | None = None):
    """Run and record locally; cloud_dir is ignored for safe backwards compatibility.

    Cloud publishing is a separate, post-milestone action, never a side-effect
    of a live Gemini run or local progress note.
    """
    if step not in ("smoke", "validation", "freeze"):
        raise ValueError(f"Unsupported step: {step}")
    private_output = private_output.expanduser().resolve()
    if not (private_output / "manifest_pre.json").is_file():
        raise ValueError("G5 preflight has not passed (missing manifest_pre.json)")
    record_activity(log_dir, "G5", f"BEGIN {step}", "STARTED",
                    f"Executing existing unmodified G5 CLI. Output: {private_output.name}/")
    sync_index(private_output, log_dir)
    command = [sys.executable, "-m", "hajj_study.g5", step, "--private-output", str(private_output)]
    proc = subprocess.Popen(command)  # Child inherits local GEMINI_API_KEY; not logged.
    try:
        while proc.poll() is None:
            sync_index(private_output, log_dir)
            time.sleep(0.3)
    except BaseException:
        proc.terminate()
        proc.wait()
        raise
    finally:
        sync_index(private_output, log_dir)
    code = proc.wait()
    if step == "freeze" and code == 0 and (private_output / "manifest_frozen.json").exists():
        stage_details = "Existing G5 freeze command returned 0 and generated manifest_frozen.json."
    else:
        stage_details = ("Existing G5 process returned exit code " + str(code) +
                         ". Real Gemini success cannot be inferred from local unit tests.")
    record_activity(log_dir, "G5", f"END {step}", "EXIT_0" if code == 0 else "FAILED",
                    stage_details)
    return code


def main(argv=None):
    p = argparse.ArgumentParser(description="Private, incremental CPCS683 execution journal.")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run unchanged G5 and auto-log each finished case")
    run.add_argument("step", choices=("smoke", "validation", "freeze"))
    scan = sub.add_parser("collect", help="Rebuild missing case ledger entries from private G5 traces")
    note = sub.add_parser("note", help="Append a manual event, such as an assistant action or test result")
    for opt in (run, scan, note):
        opt.add_argument("--private-output", default="artifacts/g5")
        opt.add_argument("--log-dir", default="artifacts/study_logs")
    note.add_argument("--stage", required=True)
    note.add_argument("--action", required=True)
    note.add_argument("--outcome", required=True)
    note.add_argument("--details", required=True)
    args = p.parse_args(argv)
    log_dir = Path(args.log_dir).expanduser().resolve()
    try:
        if args.cmd == "run":
            code = run_g5(args.step, Path(args.private_output), log_dir)
        elif args.cmd == "collect":
            new, total = sync_index(Path(args.private_output).resolve(), log_dir)
            record_activity(log_dir, "G5", "reconcile private case records", "RECORDED",
                            f"Indexed {new} new records; {total} total. No Gemini calls performed.")
            print(f"Indexed {new} new case records; {total} total (not evidence of new Gemini calls)")
            code = 0
        else:
            record_activity(log_dir, args.stage, args.action, args.outcome, args.details)
            print("Recorded progress note locally (no Gemini calls performed)")
            code = 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        p.exit(2, f"Study logging error: {exc}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
