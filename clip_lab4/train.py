# clip_lab4/train.py

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from .config import TrainConfig
from .dataset import CocoClipDataset
from .model import ImageEncoder, ClipLoss

from torch.utils.data import DataLoader, Subset
import numpy as np


data_root = "/content/coco2014"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig()

    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device

    os.makedirs(cfg.save_dir, exist_ok=True)

    train_images = os.path.join(cfg.data_root, "train2014")
    train_caps = os.path.join(cfg.data_root, cfg.train_captions)

    dataset = CocoClipDataset(
        images_dir=train_images,
        captions_json=train_caps,
        image_size=cfg.image_size,
        device=cfg.device,
        cache_text=False,  # set True later if you want precomputed text embeddings
    )

    if cfg.max_samples is not None:
        n = min(cfg.max_samples, len(dataset))
        indices = np.random.choice(len(dataset), n, replace=False)
        dataset = Subset(dataset, indices)
        print(f"Using subset of {n} samples for training.")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = ImageEncoder(embed_dim=512).to(device)
    criterion = ClipLoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for images, text_embeds in pbar:
            images = images.to(device)
            text_embeds = text_embeds.to(device)

            optimizer.zero_grad()
            img_embeds = model(images)
            loss = criterion(img_embeds, text_embeds)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % cfg.log_interval == 0:
                pbar.set_postfix(loss=running_loss / cfg.log_interval)
                running_loss = 0.0

        # simple checkpoint
        ckpt_path = os.path.join(cfg.save_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
