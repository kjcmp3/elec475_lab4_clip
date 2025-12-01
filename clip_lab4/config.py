# clip_lab4/config.py

from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_root: str = "/path/to/coco2014"   # change locally / via CLI
    train_captions: str = "annotations/captions_train2014.json"
    val_captions: str = "annotations/captions_val2014.json"
    image_size: int = 224

    batch_size: int = 64
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    max_samples: int | None = 5000

    device: str = "cuda"
    log_interval: int = 50
    save_dir: str = "weights"
