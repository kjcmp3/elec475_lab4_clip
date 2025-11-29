# clip_lab4/eval.py

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .dataset import CocoClipDataset
from .model import ImageEncoder


def cosine_sim(a, b):
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return a @ b.t()


def recall_at_k(sim_matrix, k: int) -> float:
    # sim_matrix: (N, N) image vs text similarities
    N = sim_matrix.size(0)
    targets = torch.arange(N).unsqueeze(0)  # (1, N)
    ranks = sim_matrix.argsort(dim=-1, descending=True)
    hits = (ranks[:, :k] == targets.T).any(dim=-1).float()
    return hits.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.data_root = args.data_root

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    val_images = os.path.join(cfg.data_root, "val2014")
    val_caps = os.path.join(cfg.data_root, cfg.val_captions)

    dataset = CocoClipDataset(
        images_dir=val_images,
        captions_json=val_caps,
        image_size=cfg.image_size,
        device=cfg.device,
        cache_text=True,  # precompute text embeddings
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = ImageEncoder(embed_dim=512).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    img_embeds_list = []
    text_embeds_list = []

    with torch.no_grad():
        for images, text_embeds in tqdm(loader, desc="Embedding val set"):
            images = images.to(device)
            text_embeds = text_embeds.to(device)
            img_embeds = model(images)

            img_embeds_list.append(img_embeds.cpu())
            text_embeds_list.append(text_embeds.cpu())

    img_embeds = torch.cat(img_embeds_list, dim=0)
    text_embeds = torch.cat(text_embeds_list, dim=0)

    sim_it = cosine_sim(img_embeds, text_embeds)  # image-to-text
    sim_ti = sim_it.t()                           # text-to-image

    for k in [1, 5, 10]:
        r_it = recall_at_k(sim_it, k)
        r_ti = recall_at_k(sim_ti, k)
        print(f"Recall@{k} (I→T): {r_it:.4f}, (T→I): {r_ti:.4f}")


if __name__ == "__main__":
    main()
