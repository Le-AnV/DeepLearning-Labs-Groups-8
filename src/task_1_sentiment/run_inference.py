from __future__ import annotations
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List
from transformers import pipeline
from src.common.io import load_yaml, write_jsonl

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg.get("model_name")
    if not model_name:
        raise ValueError("Missing 'model_name' in config.")

    clf = pipeline("sentiment-analysis", model=model_name)

    if args.text:
        texts = [args.text]
    else:
        texts = list(cfg.get("sample_texts", []))
        if not texts:
            raise ValueError("No input text provided. Use --text or set sample_texts in config.")

    outputs = clf(texts)

    results: List[Dict[str, Any]] = []
    for t, out in zip(texts, outputs):
        row = {
            "text": t,
            "label": out["label"],
            "score": float(out["score"]),
            "model": model_name,
            "ts": utc_now_iso(),
        }
        results.append(row)
        print(f'Text: "{t}"')
        print(f'  -> label={row["label"]}, score={row["score"]:.4f}\n')

    if args.save:
        out_path = cfg.get("output_jsonl", "outputs/task_1_inference/samples.jsonl")
        write_jsonl(out_path, results)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
