"""Encode images and captions with an OpenCLIP model into L2-normalised embeddings."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ImageDataset


def load_clip(name="ViT-B-32", pretrained="openai", device=None):
    import open_clip
    # The original OpenAI weights were trained with QuickGELU; loading them into the plain
    # architecture silently degrades the embeddings (open_clip only warns). Use the matching config.
    if pretrained == "openai" and not name.endswith("-quickgelu"):
        name += "-quickgelu"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(name)
    return model.to(device).eval(), preprocess, tokenizer, device


@torch.no_grad()
def encode_images(model, preprocess, records, root, device, batch_size=128, num_workers=4):
    loader = DataLoader(ImageDataset(records, root, preprocess), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=(device == "cuda"))
    out = []
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = model.encode_image(batch.to(device, non_blocking=True))
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=512):
    out = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i + batch_size]).to(device)   # truncates to CLIP's 77 tokens
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = model.encode_text(tokens)
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out)