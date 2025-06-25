import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import kagglehub
import pandas as pd
import torch
from PIL import Image
from pydantic import BaseModel, HttpUrl

from chest_segment.utils import download_images, is_empty, root_dir


class ChestDataloaderConfig(BaseModel):
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


class ChestDatasetConfig(BaseModel):
    root: Path = root_dir() / "data"
    target: str = "yoctoman/shcxr-lung-mask"
    source: HttpUrl = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/CXR_png/"
    split_path: Path = root_dir() / "data" / "split.csv"
    dataloader_config: ChestDataloaderConfig = ChestDataloaderConfig()
    return_dict: bool = False
    split_size: tuple[float, float, float] = (0.8, 0.1, 0.1)


class ChestDataset(torch.utils.data.Dataset):
    def __init__(self, config: ChestDatasetConfig | None = None):
        self.config = config or ChestDatasetConfig()
        if not self.config.root.exists():
            self.config.root.mkdir(parents=True, exist_ok=True)
        self.data = []
        self.split_df = None
        self.image_transforms = None
        self.mask_transforms = None
        self._load_data()

    def from_split(
        self,
        split: Literal["train", "val", "test"],
        config: ChestDatasetConfig | None = None,
    ) -> torch.utils.data.Dataset:
        dataset = ChestDataset(config=config)
        dataset.set_split(split)
        dataset.set_transforms(
            self.image_transforms, self.mask_transforms, self.all_transforms
        )
        return dataset

    def set_split(self, split: Literal["train", "val", "test"]):
        self.split_df = self.split_df[self.split_df["split"] == split]
        self.data = [self.data[i] for i in self.split_df["idx"]]

    def _load_data(self):
        if is_empty(self.config.root):
            self._download()
        for file in (self.config.root / "masks").glob("*.png"):
            image_path = (
                self.config.root
                / "images"
                / file.name.replace("_mask.png", ".png")
            )
            self.data.append(
                {
                    "mask_path": file,
                    "image_path": image_path,
                }
            )
        if not self.config.split_path.exists():
            self.split_df = self._create_train_test_split(
                *self.config.split_size
            )
            self.split_df.to_csv(self.config.split_path, index=False)
        else:
            self.split_df = pd.read_csv(self.config.split_path)

    def __len__(self):
        return len(self.data)

    def set_transforms(
        self,
        image_transforms: Callable | None = None,
        mask_transforms: Callable | None = None,
        all_transforms: Callable | None = None,
    ):
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.all_transforms = all_transforms

    def __getitem__(self, idx: int) -> dict:
        data = self.data[idx]
        image = Image.open(data["image_path"]).convert("L")
        mask = Image.open(data["mask_path"]).convert("L")
        if self.image_transforms:
            image = self.image_transforms(image)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        if self.all_transforms:
            aug = self.all_transforms(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        image = image.float()
        mask = mask.long()
        if self.config.return_dict:
            return {
                "idx": idx,
                "mask_path": data["mask_path"],
                "image_path": data["image_path"],
                "mask": mask,
                "image": image,
            }
        return image, mask

    def get_raw_image(self, idx: int) -> Image.Image:
        return Image.open(self.data[idx]["image_path"]).convert("L")

    def get_raw_mask(self, idx: int) -> Image.Image:
        return Image.open(self.data[idx]["mask_path"]).convert("L")

    def _download(self) -> Path:
        path = Path(kagglehub.dataset_download(self.config.target))
        mask_path = path / "mask"
        mask_root = self.config.root / "masks"
        mask_root.mkdir(parents=True, exist_ok=True)
        image_root = self.config.root / "images"
        image_root.mkdir(parents=True, exist_ok=True)
        urls_with_paths = []
        for file in mask_path.glob("*.png"):
            shutil.copy2(file, mask_root / file.name)
            file_name = file.name.replace("_mask.png", ".png")
            urls_with_paths.append(
                (self.config.source + file_name, image_root / file_name)
            )
        download_images(urls_with_paths)
        return path

    def _create_train_test_split(
        self, train_size: float, val_size: float, test_size: float
    ) -> pd.DataFrame:
        assert train_size + val_size + test_size == 1.0
        train_size = int(len(self.data) * train_size)
        val_size = int(len(self.data) * val_size)
        test_size = len(self.data) - train_size - val_size
        train_dataset, val_dataset, test_dataset = (
            torch.utils.data.random_split(
                range(len(self.data)),
                [train_size, val_size, test_size],
            )
        )
        data = pd.DataFrame(
            [
                {
                    "idx": idx,
                    "mask_path": i["mask_path"],
                    "image_path": i["image_path"],
                }
                for idx, i in enumerate(self.data)
            ]
        )
        data["split"] = "train"
        data.loc[list(train_dataset), "split"] = "train"
        data.loc[list(val_dataset), "split"] = "val"
        data.loc[list(test_dataset), "split"] = "test"
        return data

    def get_dataloader(self, config: ChestDataloaderConfig | None = None):
        config = config or self.config.dataloader_config
        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        )
        return dataloader


def calc_mean_std(
    dataset: ChestDataset, split: str = "train"
) -> tuple[float, float]:
    dataset = dataset.from_split(split=split, config=dataset.config)
    imgs = torch.stack([img for img, _ in dataset])
    return float(imgs.mean()), float(imgs.std())


def get_dataloaders(
    dataset: ChestDataset, config: ChestDatasetConfig | None = None
) -> list[torch.utils.data.DataLoader]:
    train_dataset = dataset.from_split(split="train", config=config)
    val_dataset = dataset.from_split(split="val", config=config)
    test_dataset = dataset.from_split(split="test", config=config)
    config = config.dataloader_config if config else ChestDataloaderConfig()
    train_loader = train_dataset.get_dataloader(config=config)
    val_loader = val_dataset.get_dataloader(config=config)
    test_loader = test_dataset.get_dataloader(config=config)
    return train_loader, val_loader, test_loader
