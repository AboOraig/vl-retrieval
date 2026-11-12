"""Final result tables (markdown) from results/*.json. Seeds of the same method are averaged (mean +- sd).

    python analysis/make_tables.py --results results --out analysis/report/results_tables.md

Methods: runs named ft_heads*, ft_last4*, ft_last8* (suffix _s1, _s2 = extra seeds) and `zeroshot`.
Inputs: results/<run>.json (RSICD test, written by train_clip.py), results/zeroshot_rsicd_test.json,
results/transfer_ucm_<run>.json, results/transfer_rsitmd_<run>.json (eval/transfer.py).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

METHODS = [("zeroshot", "Zero-shot CLIP (no training)"), ("ft_heads", "Fine-tune projections only"),
           ("ft_last4", "Fine-tune last 4 blocks"), ("ft_last8", "Fine-tune last 8 blocks")]
COLS = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10", "mR"]
CLEAN, SEEN = "not_in_rsicd_trainval", "in_rsicd_trainval"


def method_of(name):
    m = re.match(r"^(ft_heads|ft_last4|ft_last8)(_s\d+)?$", name)
    return m.group(1) if m else ("zeroshot" if name == "zeroshot" else None)


def fmt(vals):
    v = np.array(vals, dtype=float)
    return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else f"{v[0]:.1f}"


def collect(results, pattern, getter):
    """-> {method: [dict, ...]} for result files matching the pattern; getter(json) -> dict or None."""
    out = {}
    for f in sorted(Path(results).glob(pattern)):
        d = json.loads(f.read_text(encoding="utf-8"))
        name = d.get("name") or f.stem
        m = method_of(name)
        g = getter(d)
        if m and g:
            out.setdefault(m, []).append(g)
    return out


def table(title, note, runs, cols=COLS, params=None):
    lines = [f"### {title}", "", note, "", "| Method | Runs | " + " | ".join(c.replace("_", " ") for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for key, label in METHODS:
        if key in runs:
            rs = runs[key]
            lines.append(f"| {label} | {len(rs)} | " + " | ".join(fmt([r[c] for r in rs]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis/report/results_tables.md")
    a = ap.parse_args()
    R = a.results

    rsicd = collect(R, "ft_*.json", lambda d: d.get("test"))
    z = Path(R, "zeroshot_rsicd_test.json")
    if z.exists():
        rsicd["zeroshot"] = [json.loads(z.read_text())["overall"]]
    parts = [table("RSICD test split (in-domain, 1,093 images, 5,465 captions)",
                   "Recall in %, mean ± sd over training seeds (sd only when more than one run).", rsicd)]

    ucm = collect(R, "transfer_ucm_*.json", lambda d: d.get("overall"))
    if ucm:
        parts.append(table("UCM-Captions test split (trained on RSICD, never seen; 210 images, 1,050 captions)",
                           "Zero-shot transfer. No near-duplicates with RSICD were found (tools/check_overlap.py). "
                           "Gallery sizes differ between datasets, so absolute values are not comparable across tables.", ucm))

    ts = collect(R, "transfer_rsitmd_*.json", lambda d: d.get("by_overlap") and {
        "mR clean": d["by_overlap"][CLEAN]["mR"], "mR seen": d["by_overlap"][SEEN]["mR"], "mR all": d["overall"]["mR"],
        "n_clean": d["by_overlap"][CLEAN]["n_images"], "n_seen": d["by_overlap"][SEEN]["n_images"]})
    if ts:
        n_clean = next(iter(ts.values()))[0]["n_clean"]; n_seen = next(iter(ts.values()))[0]["n_seen"]
        parts.append(table(f"RSITMD test split (trained on RSICD; 452 images)",
                           f"Most RSITMD test images also exist in RSICD. Queries split by that: **{n_clean} images with no near-duplicate in "
                           f"RSICD train/val (the only valid transfer measurement)** and {n_seen} images that do (not generalization). "
                           "All queries are ranked against the full 452-image gallery. Small clean group: noisy.",
                           ts, cols=["mR clean", "mR seen", "mR all"]))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts)); print(f"saved {out}")


if __name__ == "__main__":
    main()
