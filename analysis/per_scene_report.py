"""Table + figure from results/per_scene_*.json (written by eval/per_scene.py).

    python analysis/per_scene_report.py --results results --out-dir analysis/report

Compares zero-shot, the 4-block model (mean over the seeds found: ft_last4*) and the 8-block model.
Per-class numbers rest on only ~30-40 test images per class, so differences of a few mR points
between two classes are not meaningful; the table shows n.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(results):
    runs = {}
    for f in sorted(Path(results).glob("per_scene_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        runs[d["name"]] = d
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-dir", default="analysis/report")
    ap.add_argument("--zeroshot", default="zeroshot")
    ap.add_argument("--small", default="ft_last4", help="prefix of the 4-block runs (seeds averaged)")
    ap.add_argument("--large", default="ft_last8")
    a = ap.parse_args()
    runs = load(a.results)
    zs, big = runs[a.zeroshot], runs[a.large]
    small = [r for n, r in runs.items() if n.startswith(a.small)]
    classes = sorted(zs["per_scene"])
    rows = []
    for c in classes:
        m4 = float(np.mean([r["per_scene"][c]["mR"] for r in small]))
        rows.append({"scene": c, "n_images": zs["per_scene"][c]["n_images"],
                     "zeroshot": zs["per_scene"][c]["mR"], "last4": m4,
                     "last8": big["per_scene"][c]["mR"]})
    for r in rows:
        r["gain"] = r["last8"] - r["zeroshot"]
    rows.sort(key=lambda r: -r["last8"])

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    lines = ["| Scene class | n test images | Zero-shot mR | 4 blocks (mean of %d) | 8 blocks mR | Gain (8 blocks - zero-shot) |" % len(small),
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scene']} | {r['n_images']} | {r['zeroshot']:.1f} | {r['last4']:.1f} | {r['last8']:.1f} | {r['gain']:+.1f} |")
    tot = {k: float(np.mean([r[k] for r in rows])) for k in ("zeroshot", "last4", "last8")}
    lines.append(f"| *macro-average over classes* | {sum(r['n_images'] for r in rows)} | {tot['zeroshot']:.1f} | {tot['last4']:.1f} | {tot['last8']:.1f} | {tot['last8']-tot['zeroshot']:+.1f} |")
    (out / "per_scene_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nunlabeled test images excluded from per-class queries: {zs.get('n_unlabeled_images')}")
    print(f"gain range: {min(r['gain'] for r in rows):+.1f} to {max(r['gain'] for r in rows):+.1f}; "
          f"classes where fine-tuning did not help: {[r['scene'] for r in rows if r['gain'] <= 0]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = rows[::-1]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(order) + 1.6))
    ax.hlines(y, [r["zeroshot"] for r in order], [r["last8"] for r in order], color="#BBBBBB", lw=1.5, zorder=1)
    ax.scatter([r["zeroshot"] for r in order], y, s=34, color="#E69F00", label="Zero-shot CLIP", zorder=2)
    ax.scatter([r["last8"] for r in order], y, s=34, color="#0072B2", label="Fine-tuned (8 blocks)", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r["scene"] for r in order], fontsize=8)
    ax.set_xlabel("mean recall mR on the RSICD test split (%)")
    ax.set_title("Retrieval quality per scene class", fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "per_scene.png", dpi=200)
    print(f"saved {out/'per_scene_table.md'} and {out/'per_scene.png'}")


if __name__ == "__main__":
    main()
