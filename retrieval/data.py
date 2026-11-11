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


class TrainPairs:
    """One (image, caption) pair per image per epoch: a random one of its captions.
    Because a batch is made of DISTINCT images with one caption each, no caption in the batch is a
    hidden positive for another image (no false negatives from the same image's other captions).
    Augmentation (aerial images have no canonical orientation): random flips and 90-degree rotations."""

    def __init__(self, records, root, transform, augment=True):
        self.records, self.root, self.transform, self.augment = records, Path(root), transform, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import random
        from PIL import Image
        r = self.records[i]
        with Image.open(self.root / r["image"]) as im:
            im = im.convert("RGB")
        if self.augment:
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
            k = random.randint(0, 3)
            if k:
                im = im.transpose((Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)[k - 1])
        return self.transform(im), random.choice(r["captions"])
