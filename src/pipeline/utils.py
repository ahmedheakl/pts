import json, os, time
from typing import Dict

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def save_json(obj: Dict, out_dir: str, prefix: str) -> str:
    ensure_dir(out_dir)
    fname = f"{prefix}_{int(time.time())}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return fpath
