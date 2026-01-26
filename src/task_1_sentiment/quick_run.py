from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.common.io import load_yaml, ensure_parent_dir, write_jsonl


def save_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    import yaml
    p = Path(path)
    ensure_parent_dir(p)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task_1_sentiment.yaml", help="Path to config")
    ap.add_argument("--text", default=None, help="Nếu không truyền sẽ hỏi interactive")
    ap.add_argument("--save", action="store_true", help="Append output vào output_jsonl")
    ap.add_argument("--show_probs", action="store_true", help="In thêm probs POS/NEG")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg.get("model_name")
    if not model_name:
        raise ValueError("Missing 'model_name' in config.")
    max_length = int(cfg.get("max_length", 128))
    output_jsonl = cfg.get("output_jsonl", "outputs/task1_inference/samples.jsonl")

    # 1) Lấy text từ CLI hoặc hỏi interactive
    text = args.text
    if not text:
        text = input("Enter text: ").strip()
    if not text:
        raise ValueError("Empty text.")
    cfg["last_text"] = text
    save_yaml(args.config, cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    ids = enc["input_ids"][0].tolist()
    mask = enc["attention_mask"][0].tolist()

    print("\n=== TOKENIZE DEMO ===")
    print("Text:", text)
    print("Tokens:", tokens)
    print("input_ids:", ids)
    print("attention_mask:", mask)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()

    id2label = model.config.id2label
    pred_id = int(torch.argmax(torch.tensor(probs)))
    pred_label = id2label[pred_id]
    pred_score = float(probs[pred_id])

    print("\n=== INFERENCE ===")
    print(f"label={pred_label}, score={pred_score:.4f}")

    if args.show_probs:
        print("probs:", {id2label[i]: float(probs[i]) for i in range(len(probs))})

    # 5) Save JSONL (append 1 record)
    if args.save:
        row = {
            "text": text,
            "label": pred_label,
            "score": pred_score,
            "model": model_name,
            "ts": utc_now_iso(),
        }
        # append: đọc file cũ rồi ghi lại sẽ phiền; ta ghi append đơn giản:
        p = Path(output_jsonl)
        ensure_parent_dir(p)
        with p.open("a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Appended: {output_jsonl}")


if __name__ == "__main__":
    main()
