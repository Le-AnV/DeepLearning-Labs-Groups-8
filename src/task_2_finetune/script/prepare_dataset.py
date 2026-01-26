from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from src.common.io import ensure_parent_dir

"""
  Normalize labels to {0,1} for binary classification.
  Supported cases:
  - numeric: {0,1}, {1,2}, {-1,1}, any two unique numeric values
  - string: "neg/pos", "negative/positive", "0/1", "false/true", "no/yes"
"""

def normalize_binary_labels(s: pd.Series) -> pd.Series:
    s0 = s.copy()

    # Numeric case
    if pd.api.types.is_numeric_dtype(s0):
        uniq = sorted(pd.unique(s0.dropna()))
        # If float but integer-like, cast safely
        if all(float(x).is_integer() for x in uniq):
            s0 = s0.astype("Int64")
            uniq = sorted(pd.unique(s0.dropna()))

        if set(uniq) == {0, 1}:
            return s0.astype(int)
        if set(uniq) == {1, 2}:
            return s0.map({1: 0, 2: 1}).astype(int)
        if set(uniq) == {-1, 1}:
            return s0.map({-1: 0, 1: 1}).astype(int)
        if len(uniq) == 2:
            # map lower->0, higher->1 (explicit, deterministic)
            return s0.map({uniq[0]: 0, uniq[1]: 1}).astype(int)

        raise ValueError(f"target is numeric but not binary. unique={uniq}")

    # String case
    s0 = s0.astype(str).str.strip().str.lower()
    mapping = {
        "neg": 0, "negative": 0, "0": 0, "false": 0, "no": 0,
        "pos": 1, "positive": 1, "1": 1, "true": 1, "yes": 1,
    }
    mapped = s0.map(mapping)

    # fallback: if exactly 2 unique strings, auto-map
    if mapped.isna().any():
        uniq = sorted(pd.unique(s0.dropna()))
        if len(uniq) == 2:
            auto_map = {uniq[0]: 0, uniq[1]: 1}
            mapped = s0.map(auto_map)
        else:
            raise ValueError(f"target is text but cannot map to binary. unique={uniq}")

    return mapped.astype(int)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="Raw train.csv from Kaggle")
    ap.add_argument("--out_csv", required=True, help="Processed CSV: text,label")
    args = ap.parse_args()

    train_path = Path(args.train_csv)
    out_path = Path(args.out_csv)

    if not train_path.exists():
        raise FileNotFoundError(f"Not found: {train_path}")

    df = pd.read_csv(train_path)

    required = {"text", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"train.csv must contain {required}, got: {list(df.columns)}")

    out = df[["text", "target"]].rename(columns={"target": "label"}).copy()

    # Drop NaN first to avoid "nan" string pollution
    out = out.dropna(subset=["text", "label"])

    # Clean text
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""]

    # Normalize labels to {0,1}
    out["label"] = normalize_binary_labels(out["label"])

    # Remove label conflicts for same text (professional hygiene)
    # If a text appears with both labels, drop it (simple & safe)
    conflict_texts = (
        out.groupby("text")["label"].nunique().reset_index()
    )
    conflict_texts = conflict_texts[conflict_texts["label"] > 1]["text"]
    if len(conflict_texts) > 0:
        out = out[~out["text"].isin(set(conflict_texts))]

    # Drop duplicates (now safe)
    out = out.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    # Final validation
    uniq_labels = sorted(out["label"].unique().tolist())
    if uniq_labels != [0, 1]:
        raise ValueError(f"After normalize, labels must be [0,1], got {uniq_labels}")

    ensure_parent_dir(out_path)
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(out))
    print("Label counts:\n", out["label"].value_counts(dropna=False))
    print("Head:\n", out.head(5))

if __name__ == "__main__":
    main()
