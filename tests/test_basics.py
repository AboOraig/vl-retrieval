"""Sanity checks for the metrics. Run: python tests/test_basics.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.metrics import evaluate, evaluate_by_group, i2t_ranks, t2i_ranks

rng = np.random.default_rng(0)
N, C = 200, 5
t2i = np.repeat(np.arange(N), C)

# 1) perfect similarity -> everything 100
sim = np.zeros((N, N * C)); sim[t2i, np.arange(N * C)] = 1.0
r = evaluate(sim, t2i); assert all(abs(v - 100) < 1e-9 for v in r.values()), r

# 2) constant similarity (all ties) -> pessimistic ties: near zero, never inflated
r = evaluate(np.ones((N, N * C)), t2i); assert r["i2t_R@10"] == 0 and r["t2i_R@10"] == 0, r

# 3) random similarity -> chance. t2i R@K ~ K/N ; i2t R@K ~ 1-(1-K/M)^C-ish ~ C*K/M = K/N
sim = rng.standard_normal((N, N * C))
r = evaluate(sim, t2i)
for k in (1, 5, 10):
    assert abs(r[f"t2i_R@{k}"] - 100 * k / N) < 2.5, (k, r)
    assert abs(r[f"i2t_R@{k}"] - 100 * k / N) < 2.5, (k, r)

# 4) hand-made example: 2 images, 3 captions (img0: caps 0,1 ; img1: cap 2)
sim = np.array([[0.1, 0.9, 0.5],     # img0: best correct (cap1) ranked first
                [0.8, 0.7, 0.2]])    # img1: correct cap2 ranked last -> rank 2
t = np.array([0, 0, 1])
assert list(i2t_ranks(sim, t)) == [0, 2]
# captions: cap0 -> img1 scores .8 > img0 .1 so img0 rank 1 ; cap1 -> rank 0 ; cap2 -> img0 .5 > .2 so rank 1
assert list(t2i_ranks(sim, t)) == [1, 0, 1]

# 5) group breakdown covers everyone and uses the full gallery
sim = np.zeros((N, N * C)); sim[t2i, np.arange(N * C)] = 1.0
g = np.where(np.arange(N) < 100, "a", "b")
rows = evaluate_by_group(sim, t2i, g)
assert rows["a"]["n_images"] == 100 and rows["b"]["n_captions"] == 500 and rows["a"]["mR"] == 100
print("all metric checks passed")
