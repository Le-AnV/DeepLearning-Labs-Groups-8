# this file help read yaml (input - output)
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable

# This function help load yaml config 
def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# This function help ensures the directory exists (creating it if missing) and returns its Path.
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# This function help ensures the parent directory of the output file exists before writing.
def ensure_parent_dir(file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

# This function help write a list of dictionaries to a JSONL file.
def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    ensure_parent_dir(p)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

