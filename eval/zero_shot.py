"""Zero-shot (no training) CLIP retrieval on one split of one indexed dataset.

    python eval/zero_shot.py --index data/rsicd/index.jsonl --split test --out results/zeroshot_rsicd_test

Writes <out>.json (overall + per-scene metrics). Also usable as the reference line for every
trained model: same metrics code, same protocol.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.data import load_index, flatten_captions
from retrieval.metrics import evaluate, evaluate_by_group


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="ViT-B-32-quickgelu")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    from retrieval.encode import load_clip, encode_images, encode_texts

    records, root = load_index(args.index, args.split)
    texts, txt_to_img = flatten_captions(records)
    print(f"{len(records)} images, {len(texts)} captions ({args.split})")

    model, preprocess, tokenizer, device = load_clip(args.model, args.pretrained)
    print(f"model {args.model}/{args.pretrained} on {device}")
    t0 = time.perf_counter()
    img = encode_images(model, preprocess, records, root, device, args.batch_size, args.workers)
    txt = encode_texts(model, tokenizer, texts, device)
    print(f"encoded in {time.perf_counter() - t0:.1f}s")

    sim = (img @ txt.T).numpy()
    result = {"model": args.model, "pretrained": args.pretrained, "index": args.index, "split": args.split,
              "n_images": len(records), "n_captions": len(texts),
              "overall": evaluate(sim, txt_to_img)}
    scenes = [r["scene"] for r in records]
    if all(s is not None for s in scenes):
        result["per_scene"] = evaluate_by_group(sim, txt_to_img, scenes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    o = result["overall"]
    print("\nimage->text  R@1 {i2t_R@1:.2f}  R@5 {i2t_R@5:.2f}  R@10 {i2t_R@10:.2f}".format(**o))
    print("text->image  R@1 {t2i_R@1:.2f}  R@5 {t2i_R@5:.2f}  R@10 {t2i_R@10:.2f}".format(**o))
    print(f"mR {o['mR']:.2f}")
    print(f"saved {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()