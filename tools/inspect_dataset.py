"""Print the structure of an annotation JSON and check images on disk. Read-only.

    python tools/inspect_dataset.py datasets/dataset_rsitmd.json datasets/RSITMD_images
"""
import json
import sys
from collections import Counter
from pathlib import Path

ann, img_dir = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(ann.read_text(encoding="utf-8"))
print("top level:", type(data).__name__, (sorted(data)[:10] if isinstance(data, dict) else f"len {len(data)}"))
entries = data["images"] if isinstance(data, dict) and "images" in data else data
print("entries:", len(entries))
print("first entry:", json.dumps(entries[0], ensure_ascii=False)[:700])
print("keys of first entry:", sorted(entries[0]))
for k in ("split", "category", "class", "label"):
    if k in entries[0]:
        print(f"{k} counts:", dict(Counter(str(e.get(k)) for e in entries)))
n_caps = Counter(len(e.get("sentences", e.get("captions", []))) for e in entries)
print("captions per image:", dict(n_caps))
exts = Counter(p.suffix.lower() for p in img_dir.rglob("*") if p.is_file())
print("files on disk:", dict(exts))
print("first files on disk:", sorted(p.name for p in img_dir.rglob("*") if p.is_file())[:5])
