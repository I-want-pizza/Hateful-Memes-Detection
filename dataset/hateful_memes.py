import json
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict, TypeAlias

from PIL import Image
from torch.utils.data import Dataset


class SampleDict(TypedDict):
    id: str
    img: str
    label: int
    text: str


Label: TypeAlias = int
Text: TypeAlias = str
ImageInput: TypeAlias = Image.Image

# Mapping split name -> list of JSONL files in the dataset directory
SPLIT_FILES = {
    "train":      ["train.jsonl"],
    "val":        ["dev_seen.jsonl", "dev_unseen.jsonl"],
    "dev_seen":   ["dev_seen.jsonl"],
    "dev_unseen": ["dev_unseen.jsonl"],
    "test":       ["test_seen.jsonl", "test_unseen.jsonl"],
    "test_seen":  ["test_seen.jsonl"],
    "test_unseen":["test_unseen.jsonl"],
}


class HatefulMemesDataset(Dataset):
    """
    Hateful Memes Dataset loader.

    Supports granular splits (dev_seen, dev_unseen, test_seen, test_unseen)
    in addition to the aggregate splits (train, val, test).
    """

    def __init__(
        self,
        data_dir_path: Path,
        split: Literal["train", "val", "dev_seen", "dev_unseen",
                        "test", "test_seen", "test_unseen"] = "train",
    ) -> None:
        self.data_dir_path = Path(data_dir_path)
        self.split = split
        self.img_dir = self.data_dir_path / "img"

        if not self.img_dir.is_dir():
            raise NotADirectoryError(f"Image directory not found: {self.img_dir}")

        if split not in SPLIT_FILES:
            raise ValueError(f"Unknown split '{split}'. Choose from: {list(SPLIT_FILES)}")

        self.samples: List[SampleDict] = []
        for json_name in SPLIT_FILES[split]:
            json_path = self.data_dir_path / json_name
            if not json_path.exists():
                raise FileNotFoundError(f"JSONL file not found: {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(SampleDict(**json.loads(line)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[ImageInput, Label, Text]:
        item = self.samples[idx]
        dummy_img = Image.new("RGB", (224, 224), color="black")

        img_path = self.data_dir_path / item["img"]
        if not img_path.exists():
            image_pil = dummy_img
        else:
            try:
                image_pil = Image.open(img_path).convert("RGB")
            except Exception:
                image_pil = dummy_img

        w, h = image_pil.size
        if w == 0 or h == 0:
            image_pil, w, h = dummy_img, 224, 224

        # Fix extreme aspect ratios (prevents ViLT patch collapse)
        aspect_ratio = w / h
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            max_dim = max(w, h)
            padded = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
            padded.paste(image_pil, ((max_dim - w) // 2, (max_dim - h) // 2))
            image_pil = padded
            w = h = max_dim

        # Fix microscopic images
        if w < 32 or h < 32:
            image_pil = image_pil.resize((32, 32), resample=Image.BILINEAR)

        # Downsample huge images to save RAM
        if w > 1200 or h > 1200:
            image_pil.thumbnail((1200, 1200), resample=Image.BICUBIC)

        return image_pil, item["label"], item["text"]
