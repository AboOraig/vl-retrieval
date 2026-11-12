"""Per-scene-class retrieval for several models on one split (RSICD test by default).

    python eval/per_scene.py --index data/rsicd/index.jsonl --split test --models ^
        zeroshot ft_heads=checkpoints/ft_heads.pt ft_last4=checkpoints/ft_last4.pt ^
        ft_last4_s1=checkpoints/ft_last4_s1.pt ft_last4_s2=checkpoints/ft_last4_s2.pt ft_last8=checkpoints/ft_last8.pt

Each model is `name` (zero-shot) or `name=checkpoint`. Writes results/per_scene_<name>.json.
Images without a scene label (RSICD's numbered files, e.g. 00001.jpg) are NOT used as queries for the
per-class numbers, but stay in the gallery, so rankings are computed against the full split.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from retrieval.data import load_index, flatten_captions
from retrieval.metrics import evaluate, evaluate_by_group

NONE = "__unlabeled__"


def per_scene_from_sim(sim, txt_to_img, scenes):
    groups = np.array([NONE if s is None else s for s in scenes])
    rows = evaluate_by_group(sim, txt_to_img, groups)
    unl = rows.pop(NONE, None)
    return rows, (unl["n_images"] if unl else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--models", nargs="+", required=True, help="name or name=checkpoint")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    from retrieval.encode import load_clip, encode_images, encode_texts
    records, root = load_index(args.index, args.split)
    texts, owner = flatten_captions(records)
    scenes = [r["scene"] for r in records]
    for spec in args.models:
        name, _, ckpt = spec.partition("=")
        model, preprocess, tokenizer, device = load_clip(args.model, args.pretrained, checkpoint=ckpt or None)
        img = encode_images(model, preprocess, records, root, device)
        txt = encode_texts(model, tokenizer, texts, device)
        sim = (img @ txt.T).numpy()
        rows, n_unl = per_scene_from_sim(sim, owner, scenes)
        out = {"name": name, "checkpoint": ckpt or None, "split": args.split, "n_images": len(records),
               "n_unlabeled_images": n_unl, "overall": evaluate(sim, owner), "per_scene": rows}
        Path(args.out_dir).mkdir(exist_ok=True)
        Path(args.out_dir, f"per_scene_{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"{name}: overall mR {out['overall']['mR']:.2f}, {len(rows)} classes, {n_unl} unlabeled images")
        del model


if __name__ == "__main__":
    main()
