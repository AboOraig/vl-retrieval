"""Leakage check: are images of a target dataset (e.g. RSITMD test) also in the RSICD training data?

    python tools/check_overlap.py --source data/rsicd/index.jsonl --target data/rsitmd/index.jsonl --target-split test --out analysis/overlap_rsitmd

Method: difference hash (dHash, 16x16 = 256 bits) of every image after grayscale + resize, so it is robust to
re-encoding (tif vs jpg) and rescaling (224/256/600 px). Each target image is also compared in its 8 rotations/
flips, because the same scene can be stored rotated. For every target image we report the smallest Hamming
distance to any source image and which source split it belongs to. Writes <out>.csv and a montage of the
closest pairs for a visual check. Look at the pictures: the distance threshold is a guide, not a verdict.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.data import load_index

HASH = 16
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(im):
    g = im.convert("L").resize((HASH + 1, HASH), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())            # 256 bits -> 32 bytes


def variants(im):
    out = []
    for flip in (False, True):
        base = im.transpose(Image.FLIP_LEFT_RIGHT) if flip else im
        for k in range(4):
            out.append(base.rotate(90 * k, expand=True))
    return out


def hash_set(records, root, with_variants):
    H = []
    for r in records:
        with Image.open(Path(root) / r["image"]) as im:
            im = im.convert("RGB")
            H.append(np.stack([dhash(v) for v in variants(im)]) if with_variants else dhash(im))
    return np.stack(H)


def min_distances(target_h, source_h, chunk=64):
    """target_h (T,8,32) uint8, source_h (S,32) -> (T,) min distance, (T,) argmin source index."""
    T = target_h.shape[0]
    best, arg = np.full(T, 10**9), np.zeros(T, dtype=int)
    for i in range(0, T, chunk):
        t = target_h[i:i + chunk]                                    # (c,8,32)
        x = t[:, :, None, :] ^ source_h[None, None, :, :]            # (c,8,S,32)
        d = POP[x].sum(-1).min(axis=1)                               # (c,S) min over the 8 variants
        best[i:i + chunk], arg[i:i + chunk] = d.min(axis=1), d.argmin(axis=1)
    return best, arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="index of the TRAINING dataset (RSICD)")
    ap.add_argument("--target", required=True, help="index of the dataset you test on")
    ap.add_argument("--target-split", default="test")
    ap.add_argument("--threshold", type=int, default=20, help="distance (of 256 bits) at or below which a pair is 'close'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src, src_root = load_index(a.source)
    tgt, tgt_root = load_index(a.target, a.target_split)
    print(f"hashing {len(src)} source and {len(tgt)} target images...")
    sh, th = hash_set(src, src_root, False), hash_set(tgt, tgt_root, True)
    dist, arg = min_distances(th, sh)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["target_image", "closest_source_image", "source_split", "distance"])
        for r, d, j in zip(tgt, dist, arg):
            w.writerow([r["image"], src[j]["image"], src[j]["split"], int(d)])
    print("distance to the closest source image (0 = identical, ~128 = unrelated):")
    print("  percentiles 0/1/5/25/50:", np.percentile(dist, [0, 1, 5, 25, 50]).astype(int).tolist())
    for thr in (0, 8, a.threshold):
        n = int((dist <= thr).sum())
        print(f"  target images with distance <= {thr:2d}: {n} ({100 * n / len(dist):.1f}%)")
    for sp in ("train", "val", "test"):
        n = sum(1 for d, j in zip(dist, arg) if d <= a.threshold and src[j]["split"] == sp)
        print(f"  ... of which closest to a source '{sp}' image: {n}")

    order = np.argsort(dist)[:12]
    cell, pad = 160, 4
    canvas = Image.new("RGB", (2 * cell + 3 * pad, len(order) * (cell + pad) + pad), "white")
    for row, i in enumerate(order):
        for col, path in enumerate((Path(tgt_root) / tgt[i]["image"], Path(src_root) / src[arg[i]]["image"])):
            with Image.open(path) as im:
                canvas.paste(im.convert("RGB").resize((cell, cell)), (pad + col * (cell + pad), pad + row * (cell + pad)))
    canvas.save(out.with_name(out.name + "_closest_pairs.png"))
    print(f"saved {out.with_suffix('.csv')} and {out.name}_closest_pairs.png (left: target, right: closest source image)")


if __name__ == "__main__":
    main()
