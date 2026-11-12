"""Cross-dataset retrieval: models trained on RSICD, evaluated on another dataset's split.

    python eval/transfer.py --index data/ucm/index.jsonl --split test --models zeroshot ft_heads=checkpoints/ft_heads.pt ...
    python eval/transfer.py --index data/rsitmd/index.jsonl --split test --overlap-csv analysis/overlap_rsitmd.csv --models ...

--overlap-csv (from tools/check_overlap.py) splits the queries into images that DO have a near-duplicate in RSICD's
train/val split (distance <= --threshold) and images that do not. Only the second group is a clean transfer test.
Both groups are ranked against the FULL gallery of the split. Writes results/transfer_<dataset>_<model>.json
and prints a summary table.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from retrieval.data import load_index, flatten_captions
from retrieval.metrics import evaluate, evaluate_by_group

SEEN, CLEAN = "in_rsicd_trainval", "not_in_rsicd_trainval"


def overlap_groups(records, csv_path, threshold):
    info = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            info[row["target_image"]] = (int(row["distance"]), row["source_split"])
    missing = [r["image"] for r in records if r["image"] not in info]
    if missing:
        raise SystemExit(f"{len(missing)} images are not in {csv_path} (wrong split?): {missing[:3]}")
    return [SEEN if info[r["image"]][0] <= threshold and info[r["image"]][1] in ("train", "val") else CLEAN
            for r in records]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--models", nargs="+", required=True, help="name or name=checkpoint")
    p.add_argument("--overlap-csv", default=None)
    p.add_argument("--threshold", type=int, default=20)
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()

    from retrieval.encode import load_clip, encode_images, encode_texts
    records, root = load_index(a.index, a.split)
    dataset = json.loads(Path(a.index).with_name(Path(a.index).stem + ".meta.json").read_text())["dataset"]
    texts, owner = flatten_captions(records)
    groups = overlap_groups(records, a.overlap_csv, a.threshold) if a.overlap_csv else None
    if groups:
        print(f"{groups.count(CLEAN)} of {len(groups)} {dataset} {a.split} images have no near-duplicate in RSICD train/val")
    scenes = [r["scene"] for r in records]

    table = []
    for spec in a.models:
        name, _, ckpt = spec.partition("=")
        model, preprocess, tokenizer, device = load_clip(a.model, a.pretrained, checkpoint=ckpt or None)
        sim = (encode_images(model, preprocess, records, root, device) @
               encode_texts(model, tokenizer, texts, device).T).numpy()
        res = {"name": name, "checkpoint": ckpt or None, "dataset": dataset, "split": a.split,
               "n_images": len(records), "overall": evaluate(sim, owner)}
        if groups:
            res["by_overlap"] = evaluate_by_group(sim, owner, np.array(groups))
        if all(s is not None for s in scenes):
            res["per_scene"] = evaluate_by_group(sim, owner, np.array(scenes))
        Path(a.out_dir).mkdir(exist_ok=True)
        Path(a.out_dir, f"transfer_{dataset}_{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        o = res["overall"]
        row = [name, o["mR"], o["i2t_R@1"], o["t2i_R@1"]]
        if groups:
            row += [res["by_overlap"][CLEAN]["mR"], res["by_overlap"][SEEN]["mR"]]
        table.append(row)
        del model

    head = ["model", "mR (all)", "i2t R@1", "t2i R@1"] + (["mR clean queries", "mR queries seen in RSICD"] if groups else [])
    print(f"\n{dataset} {a.split} ({len(records)} images, {len(texts)} captions)")
    print(" | ".join(head))
    for r in table:
        print(" | ".join([r[0]] + [f"{v:.2f}" for v in r[1:]]))


if __name__ == "__main__":
    main()
