"""Contrastive fine-tuning of CLIP (symmetric InfoNCE) for image-text retrieval on remote sensing.

    # quick VRAM / speed check first (a few steps, nothing saved):
    python train/train_clip.py --index data/rsicd/index.jsonl --smoke

    # real runs (names matter: results/<name>.json and checkpoints/<name>.pt)
    python train/train_clip.py --index data/rsicd/index.jsonl --name ft_last4 --unfreeze-last 4
    python train/train_clip.py --index data/rsicd/index.jsonl --name ft_heads --unfreeze-last 0   # only projections
    python train/train_clip.py --index data/rsicd/index.jsonl --name ft_full  --full              # may not fit in 4 GB

Method: dual encoder (OpenCLIP ViT-B/32, OpenAI weights). Per step a batch of B DISTINCT images, one random
caption each; logits = exp(t) * img @ txt^T; loss = (CE rows + CE columns) / 2; t learned (clamped at 100).
AdamW, linear warm-up + cosine decay, fp16 autocast, gradient clipping. The VALIDATION split picks the
best epoch (mean recall, 6 numbers); the TEST split is evaluated once, at the end, with that epoch.
Everything unfrozen starts from the pretrained weights; --unfreeze-last N unfreezes the last N transformer
blocks of both towers plus the projection layers and the temperature.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from retrieval.data import load_index, flatten_captions, TrainPairs
from retrieval.encode import load_clip, encode_images, encode_texts
from retrieval.metrics import evaluate, evaluate_by_group


def set_trainable(model, unfreeze_last, full):
    for p in model.parameters():
        p.requires_grad = full
    if full:
        return
    model.logit_scale.requires_grad = True
    for t in (model.text_projection, model.visual.proj):
        for p in ([t] if isinstance(t, torch.nn.Parameter) else t.parameters()):
            p.requires_grad = True
    for m in (model.visual.ln_post, model.ln_final):
        for p in m.parameters():
            p.requires_grad = True
    if unfreeze_last > 0:
        for blocks in (model.visual.transformer.resblocks, model.transformer.resblocks):
            for b in list(blocks)[-unfreeze_last:]:
                for p in b.parameters():
                    p.requires_grad = True


def make_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or "logit_scale" in name else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.98), eps=1e-6)


def lr_at(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def retrieval_eval(model, preprocess, tokenizer, records, root, device, per_scene=False):
    model.eval()
    texts, owner = flatten_captions(records)
    img = encode_images(model, preprocess, records, root, device, batch_size=128, num_workers=4)
    txt = encode_texts(model, tokenizer, texts, device)
    sim = (img @ txt.T).numpy()
    out = evaluate(sim, owner)
    scenes = [r["scene"] for r in records]
    if per_scene and all(s is not None for s in scenes):
        return out, evaluate_by_group(sim, owner, scenes)
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--name", default="run")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--unfreeze-last", type=int, default=4)
    ap.add_argument("--full", action="store_true", help="train every parameter")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=4, help="stop after N epochs without val improvement")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="5 steps + one val pass on 128 images; saves nothing")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model, preprocess, tokenizer, device = load_clip(args.model, args.pretrained)
    amp = (device == "cuda") and not args.no_amp
    set_trainable(model, args.unfreeze_last, args.full)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"device {device}, amp {amp}, trainable params {n_train/1e6:.1f}M / {n_all/1e6:.1f}M")

    train_rec, root = load_index(args.index, "train")
    val_rec, _ = load_index(args.index, "val")
    test_rec, _ = load_index(args.index, "test")
    if args.smoke:
        train_rec, val_rec = train_rec[:args.batch_size * 5], val_rec[:128]

    loader = DataLoader(TrainPairs(train_rec, root, preprocess, augment=not args.no_augment),
                        batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=(device == "cuda"),
                        persistent_workers=args.workers > 0)
    steps_per_epoch = len(loader)
    epochs = 1 if args.smoke else args.epochs
    total = steps_per_epoch * epochs
    opt = make_optimizer(model, args.lr, args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, total, max(1, int(args.warmup_frac * total))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    trainable = [p for p in model.parameters() if p.requires_grad]
    labels = torch.arange(args.batch_size, device=device)

    ckpt_path = Path("checkpoints") / f"{args.name}.pt"
    history, best, best_epoch, bad = [], -1.0, -1, 0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running, t0 = [], time.perf_counter()
        for step, (imgs, caps) in enumerate(loader, 1):
            imgs, tokens = imgs.to(device, non_blocking=True), tokenizer(list(caps)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                fi = model.encode_image(imgs)
                ft = model.encode_text(tokens)
            fi, ft = F.normalize(fi.float(), dim=-1), F.normalize(ft.float(), dim=-1)
            logits = model.logit_scale.exp() * fi @ ft.T
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
            running.append(loss.item())
            if step % 20 == 0 or step == steps_per_epoch:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {np.mean(running[-20:]):.4f}", flush=True)
        val, _ = retrieval_eval(model, preprocess, tokenizer, val_rec, root, device)
        rec = {"epoch": epoch, "train_loss": float(np.mean(running)), "val_mR": val["mR"],
               "val": val, "temperature": float(model.logit_scale.exp()),
               "epoch_seconds": time.perf_counter() - t0}
        history.append(rec)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0
        print(f"epoch {epoch}: loss {rec['train_loss']:.4f}  val mR {val['mR']:.2f}  "
              f"(i2t R@1 {val['i2t_R@1']:.1f}, t2i R@1 {val['t2i_R@1']:.1f})  "
              f"scale {rec['temperature']:.1f}  {rec['epoch_seconds']:.0f}s  peak VRAM {peak:.2f} GiB", flush=True)
        if args.smoke:
            print("smoke test finished (nothing saved)."); return
        if val["mR"] > best:
            best, best_epoch, bad = val["mR"], epoch, 0
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no validation improvement for {args.patience} epochs"); break

    print(f"\nbest epoch {best_epoch} (val mR {best:.2f}); evaluating TEST once with it")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")); model.to(device)
    test, per_scene = retrieval_eval(model, preprocess, tokenizer, test_rec, root, device, per_scene=True)
    result = {"name": args.name, "args": vars(args), "best_epoch": best_epoch, "best_val_mR": best,
              "trainable_params_M": n_train / 1e6, "train_minutes": (time.perf_counter() - t_start) / 60,
              "test": test, "per_scene": per_scene, "history": history}
    Path("results").mkdir(exist_ok=True)
    Path("results", f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("TEST  image->text R@1 {i2t_R@1:.2f} R@5 {i2t_R@5:.2f} R@10 {i2t_R@10:.2f}".format(**test))
    print("TEST  text->image R@1 {t2i_R@1:.2f} R@5 {t2i_R@5:.2f} R@10 {t2i_R@10:.2f}   mR {mR:.2f}".format(**test))
    print(f"saved results/{args.name}.json and {ckpt_path}")


if __name__ == "__main__":
    main()
