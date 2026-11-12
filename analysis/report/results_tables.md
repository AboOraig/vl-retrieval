### RSICD test split (in-domain, 1,093 images, 5,465 captions)

Recall in %, mean ± sd over training seeds (sd only when more than one run).

| Method | Runs | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | mR |
|---|---|---|---|---|---|---|---|---|
| Zero-shot CLIP (no training) | 1 | 4.9 | 14.4 | 23.0 | 5.5 | 17.6 | 27.8 | 15.5 |
| Fine-tune projections only | 1 | 8.5 | 22.7 | 34.4 | 7.9 | 26.4 | 41.1 | 23.5 |
| Fine-tune last 4 blocks | 3 | 13.0 ± 0.7 | 30.8 ± 1.3 | 44.8 ± 1.7 | 12.1 ± 0.5 | 33.2 ± 0.2 | 50.1 ± 0.9 | 30.7 ± 0.6 |
| Fine-tune last 8 blocks | 3 | 16.0 ± 1.0 | 33.5 ± 2.7 | 45.9 ± 1.9 | 13.0 ± 0.6 | 36.6 ± 1.0 | 53.2 ± 1.1 | 33.0 ± 1.3 |

### UCM-Captions test split (trained on RSICD, never seen; 210 images, 1,050 captions)

Zero-shot transfer. No near-duplicates with RSICD were found (tools/check_overlap.py). Gallery sizes differ between datasets, so absolute values are not comparable across tables.

| Method | Runs | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | mR |
|---|---|---|---|---|---|---|---|---|
| Zero-shot CLIP (no training) | 1 | 6.2 | 22.9 | 46.7 | 8.7 | 36.4 | 60.3 | 30.2 |
| Fine-tune projections only | 1 | 6.7 | 21.9 | 53.3 | 10.8 | 43.1 | 72.5 | 34.7 |
| Fine-tune last 4 blocks | 3 | 8.9 ± 0.3 | 24.0 ± 0.3 | 58.6 ± 2.9 | 12.8 ± 0.7 | 47.1 ± 0.4 | 75.0 ± 0.8 | 37.7 ± 0.6 |
| Fine-tune last 8 blocks | 3 | 8.4 ± 1.8 | 24.4 ± 2.4 | 56.8 ± 3.3 | 13.5 ± 0.8 | 47.0 ± 1.8 | 74.6 ± 2.4 | 37.5 ± 1.2 |

### RSITMD test split (trained on RSICD; 452 images)

Most RSITMD test images also exist in RSICD. Queries split by that: **91 images with no near-duplicate in RSICD train/val (the only valid transfer measurement)** and 361 images that do (not generalization). All queries are ranked against the full 452-image gallery. Small clean group: noisy.

| Method | Runs | mR clean | mR seen | mR all |
|---|---|---|---|---|
| Zero-shot CLIP (no training) | 1 | 29.6 | 22.8 | 24.2 |
| Fine-tune projections only | 1 | 39.2 | 36.3 | 36.9 |
| Fine-tune last 4 blocks | 3 | 45.5 ± 1.6 | 50.8 ± 5.5 | 49.7 ± 4.7 |
| Fine-tune last 8 blocks | 3 | 48.4 ± 1.9 | 55.4 ± 7.9 | 54.0 ± 6.5 |
