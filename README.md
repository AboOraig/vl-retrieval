# Vision-Language Retrieval for Remote Sensing

Contrastive image-text alignment of remote-sensing images and captions in a shared embedding space,
evaluated for image-to-text and text-to-image retrieval, within and across datasets.

## Layout
```
data/        dataset preparation scripts (+ generated indexes, ignored by git)
retrieval/   index loading, encoding, retrieval metrics
eval/        evaluation scripts (zero-shot baseline, later: fine-tuned models)
train/       contrastive training (step 3)
analysis/    tables and figures
tests/       quick checks:  python tests/test_basics.py
```

## Setup
```
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```
The development GPU has 4 GB of memory: inference is comfortable, training uses mixed precision and
small batches (see step 3).

## Step 1: data and zero-shot baseline
1. Download RSICD from https://github.com/201528014227051/RSICD_optimal (the README lists Google Drive,
   Baidu and MEGA mirrors; the files are `RSICD_images.zip` and `dataset_rsicd.json`).
   Unzip the images next to the json, e.g. `\datasets\RSICD\`.
2. Build the index (validates the annotation format and prints dataset statistics):
```
python data/prepare_rsicd.py --root D:\datasets\RSICD --out data/rsicd/index.jsonl
```
3. Zero-shot CLIP on the RSICD test split:
```
python eval/zero_shot.py --index data/rsicd/index.jsonl --split test --out results/zeroshot_rsicd_test
```
Protocol: Recall@1/5/10 in both directions and their mean (mR). An image-to-text query is correct if any
of the image's 5 captions is in the top K; a text-to-image query is correct if its image is in the top K.
