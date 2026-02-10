"""PyTorch Lightning data module for MrOS fracture prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import pytorch_lightning as pl
from monai.data import DataLoader, ImageDataset

from osteo_fracture.config import DataConfig
from osteo_fracture.data.transforms import FractureTransforms


class FractureDataModule(pl.LightningDataModule):
    """Load split-based MrOS CT patch datasets."""

    def __init__(self, data_cfg: DataConfig, project_root: Path) -> None:
        super().__init__()
        self.cfg = data_cfg
        self.img_dir: Final[Path] = project_root / data_cfg.img_dir
        self.labels_path: Final[Path] = project_root / data_cfg.labels_file
        self.train_set: ImageDataset | None = None
        self.val_set: ImageDataset | None = None
        self.test_set: ImageDataset | None = None

    def _load_labels(self) -> pd.DataFrame:
        return pd.read_csv(self.labels_path, sep=";")

    def _build_split(self, df: pd.DataFrame, split_name: str) -> tuple[list[str], list[int]]:
        split_df = df[df[self.cfg.split_name] == split_name]
        filenames = split_df.Filename.tolist()
        available = {p.name for p in self.img_dir.iterdir()}
        images = sorted(list(set(filenames).intersection(available)))
        image_paths = [str(self.img_dir / img) for img in images]
        labels = [int(split_df.loc[split_df["Filename"] == img][self.cfg.label].values[0]) for img in images]
        return image_paths, labels

    def get_class_ratio(self) -> float:
        df = self._load_labels()
        train = df[df[self.cfg.split_name] == "Training"]
        positives = max(1, len(train[train[self.cfg.label] == 1]))
        negatives = len(train[train[self.cfg.label] == 0])
        return negatives / positives

    def setup(self, stage: str | None = None) -> None:
        df = self._load_labels()
        train = self._build_split(df, "Training")
        val = self._build_split(df, "Validation")
        test = self._build_split(df, "Testing")

        self.train_set = ImageDataset(
            image_files=train[0],
            labels=train[1],
            transform=FractureTransforms(
                mode="training",
                input_size=self.cfg.input_size,
                input_dimension=self.cfg.input_dimension,
                image_augmentation_prob=self.cfg.image_augmentation_prob,
                augmentation_magnitude=self.cfg.augmentation_magnitude,
                num_sequential_transforms=self.cfg.num_sequential_transforms,
            ),
        )
        self.val_set = ImageDataset(
            image_files=val[0],
            labels=val[1],
            transform=FractureTransforms(
                mode="validation",
                input_size=self.cfg.input_size,
                input_dimension=self.cfg.input_dimension,
                image_augmentation_prob=self.cfg.image_augmentation_prob,
                augmentation_magnitude=self.cfg.augmentation_magnitude,
                num_sequential_transforms=self.cfg.num_sequential_transforms,
            ),
        )
        self.test_set = ImageDataset(
            image_files=test[0],
            labels=test[1],
            transform=FractureTransforms(
                mode="testing",
                input_size=self.cfg.input_size,
                input_dimension=self.cfg.input_dimension,
                image_augmentation_prob=self.cfg.image_augmentation_prob,
                augmentation_magnitude=self.cfg.augmentation_magnitude,
                num_sequential_transforms=self.cfg.num_sequential_transforms,
            ),
        )

    def _loader(self, dataset: ImageDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_set is not None
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_set is not None
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_set is not None
        return self._loader(self.test_set, shuffle=False)
