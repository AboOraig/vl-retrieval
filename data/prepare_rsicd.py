"""Build a clean index of RSICD: one JSON line per image (path, split, captions, scene class).

Usage:
    python data/prepare_rsicd.py --root D:\\datasets\\RSICD --out data/rsicd/index.jsonl

--root must contain `dataset_rsicd.json` (override with --annotations) and the images, in any
sub-folder (they are found by file name). The expected annotation format is the common
Karpathy-style JSON:
    {"images": [{"filename": "airport_1.jpg", "split": "train",
                 "sentences": [{"raw": "..."}, ...]}, ...]}
If your file differs, the script stops and prints the keys it found instead of guessing.

The official split in the file is used (comparable with published results). --random-split exists
only as a last resort when the file has no split field and its numbers are NOT comparable.
The scene class is derived from the file name (airport_1.jpg -> airport) when that pattern holds.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
FILENAME_KEYS = ("filename", "file_name", "image", "img", "img_path", "image_name")
SPLITS = {"train": "train", "restval": "train", "val": "val", "valid": "val", "validation": "val",
          "dev": "val", "test": "test"}
SCENE_PATTERN = re.compile(r"^(.+?)_\d+\.[A-Za-z]+$")


def fail(message):
    raise SystemExit("ERROR: " + message)


def caption_text(item):
    """One caption from a str, or a dict with raw / caption / sentence / text / tokens."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("raw", "caption", "sentence", "text"):
            if isinstance(item.get(key), str):
                return item[key].strip()
        if isinstance(item.get("tokens"), list):
            return " ".join(item["tokens"]).strip()
    return None


def parse_annotations(data):
    """-> list of (filename, split or None, [captions]). Stops with a clear message if unknown."""
    entries = data["images"] if isinstance(data, dict) and "images" in data else data
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        fail("unexpected annotation format: expected a list of images (or a dict with an 'images' list). "
             f"Top level is a {type(data).__name__}"
             + (f" with keys {sorted(data)[:10]}" if isinstance(data, dict) else "") + ".")
    parsed = []
    for e in entries:
        fname = next((e[k] for k in FILENAME_KEYS if isinstance(e.get(k), str)), None)
        raw = e.get("sentences", e.get("captions", e.get("caption")))
        if isinstance(raw, str):
            raw = [raw]
        captions = [c for c in (caption_text(x) for x in (raw or [])) if c]
        if fname is None or not captions:
            fail("could not read an image entry. Keys of the first entry: "
                 f"{sorted(entries[0])}. First entry (truncated): {str(entries[0])[:300]}")
        split = SPLITS.get(str(e.get("split", "")).lower())
        parsed.append((Path(fname).name, split, captions))
    return parsed


def find_images(root):
    found = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.name not in found:
            found[p.name] = p
    return found


def scene_of(fname):
    m = SCENE_PATTERN.match(fname)
    return m.group(1) if m else None


def build_index(root, annotations, out, random_split=False, seed=0):
    root, out = Path(root).resolve(), Path(out)
    ann_path = Path(annotations) if annotations else root / "dataset_rsicd.json"
    if not ann_path.exists():
        fail(f"annotation file not found: {ann_path}. Pass --annotations.")
    parsed = parse_annotations(json.loads(ann_path.read_text(encoding="utf-8")))

    images = find_images(root)
    missing = [f for f, _, _ in parsed if f not in images]
    if missing:
        fail(f"{len(missing)} of {len(parsed)} annotated images were not found under {root} "
             f"(first missing: {missing[:5]}). Did you unzip the images?")

    if any(s is None for _, s, _ in parsed):
        if not random_split:
            fail("the annotations have no (usable) 'split' field. Official splits are needed for "
                 "comparable results; use --random-split only if you accept non-comparable numbers.")
        rng = random.Random(seed)
        order = list(range(len(parsed)))
        rng.shuffle(order)
        cut1, cut2 = int(0.8 * len(order)), int(0.9 * len(order))
        names = {i: ("train" if r < cut1 else "val" if r < cut2 else "test") for r, i in enumerate(order)}
        parsed = [(f, names[i], c) for i, (f, _, c) in enumerate(parsed)]
        print("WARNING: random 80/10/10 split used - NOT comparable with published results.")

    scenes = [scene_of(f) for f, _, _ in parsed]
    if sum(s is not None for s in scenes) < 0.9 * len(scenes):
        print("WARNING: file names do not follow '<scene>_<number>.<ext>'; scene classes unavailable.")
        scenes = [None] * len(parsed)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for (fname, split, captions), scene in zip(parsed, scenes):
            rel = images[fname].relative_to(root).as_posix()
            fh.write(json.dumps({"image": rel, "split": split, "captions": captions, "scene": scene},
                                ensure_ascii=False) + "\n")
    meta = {"root": str(root), "dataset": "rsicd", "n_images": len(parsed), "annotations": ann_path.name}
    out.with_name(out.stem + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    n_caps = Counter(len(c) for _, _, c in parsed)
    print(f"{len(parsed)} images  |  splits: {dict(Counter(s for _, s, _ in parsed))}")
    print(f"captions per image: {dict(sorted(n_caps.items()))}")
    cls = Counter(s for s in scenes if s)
    if cls:
        print(f"{len(cls)} scene classes: " + ", ".join(f"{k}({v})" for k, v in sorted(cls.items())))
    if len(parsed) != 10921:
        print(f"NOTE: the RSICD release has 10,921 images; this index has {len(parsed)}.")
    if set(n_caps) != {5}:
        print("NOTE: RSICD is documented as 5 captions per image; check the distribution above.")
    print(f"wrote {out}")
    return parsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder with dataset_rsicd.json and the images")
    p.add_argument("--annotations", default=None, help="annotation json (default: <root>/dataset_rsicd.json)")
    p.add_argument("--out", default="data/rsicd/index.jsonl")
    p.add_argument("--random-split", action="store_true", help="last resort, NOT comparable with papers")
    args = p.parse_args()
    build_index(args.root, args.annotations, args.out, args.random_split)


if __name__ == "__main__":
    main()
