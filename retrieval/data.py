"""Index loading and torch datasets shared by evaluation and training."""
import json
from pathlib import Path


def load_index(index_path, split=None):
    """-> (records, root). Each record: image (relative path), split, captions, scene."""
    index_path = Path(index_path)
    meta = json.loads(index_path.with_name(index_path.stem + ".meta.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    if split:
        records = [r for r in records if r["split"] == split]
        if not records:
            raise SystemExit(f"no images with split '{split}' in {index_path}")
    return records, Path(meta["root"])


def flatten_captions(records):
    """-> (captions list, txt_to_img array): caption j describes records[txt_to_img[j]]."""
    texts, owner = [], []
    for i, r in enumerate(records):
        for c in r["captions"]:
            texts.append(c)
            owner.append(i)
    return texts, owner


class ImageDataset:
    """Loads RGB images and applies a transform. (Plain class: works with torch DataLoader.)"""

    def __init__(self, records, root, transform):
        self.paths = [Path(root) / r["image"] for r in records]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB"))
