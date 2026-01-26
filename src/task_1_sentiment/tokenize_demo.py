from __future__ import annotations
import argparse
from transformers import AutoTokenizer
from src.common.io import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg.get("model_name")
    if not model_name:
        raise ValueError("Missing 'model_name' in config.")

    max_length = int(cfg.get("max_length", 128))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    enc = tokenizer(
        args.text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )

    tokens = tokenizer.tokenize(args.text)

    print(f"Model: {model_name}")
    print(f"Text: {args.text}\n")

    print("Tokens:")
    print(tokens, "\n")

    print("input_ids:")
    print(enc["input_ids"], "\n")

    print("attention_mask:")
    print(enc["attention_mask"], "\n")


if __name__ == "__main__":
    main()
