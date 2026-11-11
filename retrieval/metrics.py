"""Cross-modal retrieval metrics (numpy only).

Setup: N images, M captions, `txt_to_img[j]` = index of the image caption j describes.
sim[i, j] = similarity of image i and caption j.

  image -> text : query = image; correct if ANY of its captions is in the top K (rank of the best one).
  text -> image : query = caption; correct if its image is in the top K.
  mR            : mean of the six recalls (R@1/5/10 in both directions), as in the RS retrieval papers.
Ties are broken pessimistically (a tied wrong item counts as ranked ahead), so constant
similarities cannot give inflated scores.
"""
import numpy as np

KS = (1, 5, 10)


def i2t_ranks(sim, txt_to_img):
    """0-based rank of the best-ranked correct caption, per image."""
    n_img = sim.shape[0]
    correct = np.zeros_like(sim, dtype=bool)
    correct[txt_to_img, np.arange(sim.shape[1])] = True
    best_correct = np.where(correct, sim, -np.inf).max(axis=1)           # best score among correct captions
    ahead = (np.where(correct, -np.inf, sim) >= best_correct[:, None]).sum(axis=1)
    return ahead


def t2i_ranks(sim, txt_to_img):
    """0-based rank of the correct image, per caption."""
    true_score = sim[txt_to_img, np.arange(sim.shape[1])]                 # (M,)
    wrong_ahead = (sim >= true_score[None, :]).sum(axis=0) - 1            # excludes itself
    return wrong_ahead


def recalls(ranks, ks=KS):
    return {f"R@{k}": 100.0 * float(np.mean(ranks < k)) for k in ks}


def evaluate(sim, txt_to_img):
    sim = np.asarray(sim, dtype=np.float32)
    txt_to_img = np.asarray(txt_to_img)
    r_i2t, r_t2i = i2t_ranks(sim, txt_to_img), t2i_ranks(sim, txt_to_img)
    out = {}
    out.update({f"i2t_{k}": v for k, v in recalls(r_i2t).items()})
    out.update({f"t2i_{k}": v for k, v in recalls(r_t2i).items()})
    out["mR"] = float(np.mean([v for v in out.values()]))
    return out


def evaluate_by_group(sim, txt_to_img, img_groups):
    """Per scene class: i2t recalls of the images in the group, t2i recalls of their captions.
    Rankings still run against the FULL gallery (the group only selects the queries)."""
    sim = np.asarray(sim, dtype=np.float32)
    txt_to_img = np.asarray(txt_to_img)
    img_groups = np.asarray(img_groups)
    r_i2t, r_t2i = i2t_ranks(sim, txt_to_img), t2i_ranks(sim, txt_to_img)
    rows = {}
    for g in sorted(set(img_groups.tolist())):
        img_mask = img_groups == g
        txt_mask = img_mask[txt_to_img]
        row = {"n_images": int(img_mask.sum()), "n_captions": int(txt_mask.sum())}
        row.update({f"i2t_{k}": v for k, v in recalls(r_i2t[img_mask]).items()})
        row.update({f"t2i_{k}": v for k, v in recalls(r_t2i[txt_mask]).items()})
        row["mR"] = float(np.mean([row[f"{d}_R@{k}"] for d in ("i2t", "t2i") for k in KS]))
        rows[g] = row
    return rows
