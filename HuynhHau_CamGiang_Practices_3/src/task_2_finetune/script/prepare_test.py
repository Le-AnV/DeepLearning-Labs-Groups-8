from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Common "null-like" strings that sometimes appear in CSVs.
_NULL_LIKE = {"", "null", "<null>", "none", "nan"}


def clean_optional_str(s: pd.Series) -> pd.Series:
    """
    Trim whitespace while preserving missing values (NA).
    Also normalizes common 'null-like' literals to NA.
    """
    s = s.astype("string").str.strip()
    return s.mask(s.str.lower().isin(_NULL_LIKE), pd.NA)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean Kaggle test.csv for inference/submission.")
    ap.add_argument("--test_csv", required=True, help="Path to raw test.csv")
    ap.add_argument("--out_csv", required=True, help="Path to write cleaned test CSV")
    args = ap.parse_args()

    test_path = Path(args.test_csv)
    out_path = Path(args.out_csv)

    if not test_path.exists():
        raise FileNotFoundError(f"Not found: {test_path}")

    df = pd.read_csv(test_path)

    # Minimal schema required for Kaggle-style submission workflows.
    for col in ("id", "text"):
        if col not in df.columns:
            raise ValueError(f"test.csv must contain '{col}', got: {list(df.columns)}")

    # Keep only relevant columns if present.
    keep_cols = [c for c in ("id", "keyword", "location", "text") if c in df.columns]
    out = df[keep_cols].copy()

    # Clean text (must be non-empty for tokenization/inference).
    out["text"] = out["text"].astype("string").str.strip()
    out = out.dropna(subset=["text"])
    out = out[out["text"] != ""]

    # Ensure id is numeric and stable for submission merge.
    out["id"] = pd.to_numeric(out["id"], errors="raise").astype(int)

    # Clean optional metadata fields (safe to keep NA).
    if "keyword" in out.columns:
        out["keyword"] = clean_optional_str(out["keyword"])
    if "location" in out.columns:
        out["location"] = clean_optional_str(out["location"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(out))
    print("Columns:", list(out.columns))
    print("Head:\n", out.head(5))


if __name__ == "__main__":
    main()
