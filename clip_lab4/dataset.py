# clip_lab4/dataset.py

import json
import os
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CocoClipDataset(Dataset):
    """
    COCO2014 image-caption dataset for CLIP-style training.
    For simplicity, this version uses the first caption per image.
    """

    def __init__(
        self,
        images_dir: str,
        captions_json: str,
        text_model_name: str = "openai/clip-vit-base-patch32",
        image_size: int = 224,
        device: str = "cuda",
        cache_text: bool = False,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.device = device
        self.cache_text = cache_text

        with open(captions_json, "r") as f:
            captions_data = json.load(f)

        # Map image_id -> first caption
        img_id_to_caption = {}
        for ann in captions_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_id_to_caption:
                img_id_to_caption[img_id] = ann["caption"]

        # Map image_id -> file_name
        img_id_to_filename = {
            img["id"]: img["file_name"]
            for img in captions_data["images"]
            if img["id"] in img_id_to_caption
        }

        self.items = []
        for img_id, caption in img_id_to_caption.items():
            if img_id in img_id_to_filename:
                self.items.append(
                    {
                        "image_path": os.path.join(images_dir, img_id_to_filename[img_id]),
                        "caption": caption,
                    }
                )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

        # Set up CLIP text encoder (frozen)
        self.tokenizer = CLIPTokenizer.from_pretrained(text_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_model_name).eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.to(self.device)

        # Optional: precompute text embeddings
        self.text_embeddings = None
        if self.cache_text:
            self._build_text_cache()

    def _encode_text(self, caption: str) -> torch.Tensor:
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
            # Use pooled output (CLS-style) or mean pooling
            text_emb = outputs.last_hidden_state[:, 0, :]  # (1, 512)
        return text_emb.squeeze(0)  # (512,)

    def _build_text_cache(self):
        print("Building text embedding cache...")
        self.text_embeddings = []
        for item in self.items:
            emb = self._encode_text(item["caption"]).cpu()
            self.text_embeddings.append(emb)
        self.text_embeddings = torch.stack(self.text_embeddings)  # (N, 512)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = self.transform(image)

        if self.text_embeddings is not None:
            text_emb = self.text_embeddings[idx]
        else:
            text_emb = self._encode_text(item["caption"]).cpu()

        return image, text_emb
