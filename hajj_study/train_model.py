"""Train and freeze one XGBoost + SMOTENC model on training rows only.

This model is shared unchanged by both experimental conditions. It does NOT
infer the reference proxy from its seven label-construction fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from hajj_study.audit_data import audit
from hajj_study.feature_contract import ALLOWED_MODEL_FEATURES, CATEGORICAL_FEATURES, validate_features
from hajj_study.proxy import risk_label

LABELS = ("LOW", "MEDIUM", "HIGH")
SEED = 42


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_and_freeze(source_csv: str | Path, splits_csv: str | Path, out_dir: str | Path) -> dict:
    source_csv, splits_csv = Path(source_csv), Path(splits_csv)
    audit(source_csv)
    df = pd.read_csv(source_csv)
    validate_features(df.columns)
    split_table = pd.read_csv(splits_csv)
    if split_table.columns.tolist() != ["source_row_index", "split", "proxy_label"]:
        raise ValueError("Invalid split manifest format")
    if len(split_table) != len(df) or not np.array_equal(split_table.source_row_index.to_numpy(), np.arange(len(df))):
        raise ValueError("Source-row IDs are not unique and ordered")
    expected = np.array([risk_label(row) for row in df.to_dict(orient="records")])
    if not np.array_equal(split_table.proxy_label.to_numpy(), expected):
        raise ValueError("Frozen split labels disagree with the frozen proxy")
    masks = {name: split_table.split.eq(name).to_numpy() for name in ("train", "validation", "test")}
    if {name: int(mask.sum()) for name, mask in masks.items()} != {"train": 7000, "validation": 1000, "test": 2000}:
        raise ValueError("Frozen split sizes differ from contract")

    numeric = [name for name in ALLOWED_MODEL_FEATURES if name not in CATEGORICAL_FEATURES]
    categorical = list(CATEGORICAL_FEATURES)
    # Fit preprocessing on training records only.
    numeric_scaler = StandardScaler().fit(df.loc[masks["train"], numeric])
    categorical_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
        df.loc[masks["train"], categorical]
    )

    def transform(mask: np.ndarray) -> np.ndarray:
        first = numeric_scaler.transform(df.loc[mask, numeric])
        second = categorical_encoder.transform(df.loc[mask, categorical])
        return np.hstack([first, second]).astype("float64")

    x_train = transform(masks["train"])
    y_all = np.array([LABELS.index(label) for label in expected])
    y_train = y_all[masks["train"]]
    categorical_positions = list(range(len(numeric), len(numeric) + len(categorical)))
    smote = SMOTENC(categorical_features=categorical_positions, random_state=SEED, k_neighbors=5)
    x_balanced, y_balanced = smote.fit_resample(x_train, y_train)

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=1.0, colsample_bytree=1.0,
        objective="multi:softprob", eval_metric="mlogloss",
        tree_method="hist", random_state=SEED, n_jobs=2,
    )
    model.fit(x_balanced, y_balanced)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_file = output / "shared_model.joblib"
    joblib.dump({
        "model": model, "numeric_scaler": numeric_scaler,
        "categorical_encoder": categorical_encoder,
        "numeric_features": numeric, "categorical_features": categorical,
        "allowed_features": list(ALLOWED_MODEL_FEATURES),
        "risk_label_order": list(LABELS),
        "split_sha256": _hash(splits_csv), "raw_csv_sha256": _hash(source_csv),
        "seed": SEED,
    }, model_file, compress=3)

    metrics = {}
    predictions = output / "predictions.jsonl"
    with predictions.open("w", encoding="utf-8", newline="\n") as stream:
        for group in ("validation", "test"):
            mask = masks[group]
            probabilities = model.predict_proba(transform(mask))
            predicted = np.argmax(probabilities, axis=1)
            gold = y_all[mask]
            majority = int(np.bincount(y_train).argmax())
            metrics[group] = {
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(gold, predicted)),
                "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
                "majority_class": LABELS[majority],
                "majority_accuracy": float(accuracy_score(gold, np.full(len(gold), majority))),
                "confusion_matrix": confusion_matrix(gold, predicted, labels=[0,1,2]).tolist(),
                "per_class": classification_report(gold, predicted, labels=[0,1,2],
                    target_names=LABELS, output_dict=True, zero_division=0),
                "predicted_class_counts": {name: int(np.sum(predicted==i)) for i,name in enumerate(LABELS)},
            }
            for source_row_index, probs in zip(np.flatnonzero(mask), probabilities):
                record = {
                    "source_row_index": int(source_row_index),
                    "split": group,
                    "predicted_risk": LABELS[int(np.argmax(probs))],
                    "probabilities": {name: round(float(value), 10) for name, value in zip(LABELS, probs)},
                }
                stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True)+"\n")

    report = {
        "raw_csv_sha256": _hash(source_csv),
        "row_splits_sha256": _hash(splits_csv),
        "model_sha256": _hash(model_file),
        "predictions_sha256": _hash(predictions),
        "train_size_before_smote": int(len(y_train)),
        "train_size_after_smote": int(len(y_balanced)),
        "allowed_features": list(ALLOWED_MODEL_FEATURES),
        "SMOTENC_fit_split": "train only",
        "versions": {
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "imblearn": imblearn.__version__, "xgboost": xgboost.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
        },
        "metrics": metrics,
        "limitation": "Shared model is evaluated on synthetic proxy labels; not real-world Hajj risk accuracy",
    }
    (output / "model_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = fit_and_freeze(args.csv, args.splits, args.out)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
