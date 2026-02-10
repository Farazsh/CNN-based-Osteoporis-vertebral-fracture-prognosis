"""Data preprocessing and augmentation transforms."""

from __future__ import annotations

import random

import monai
import numpy as np
import torchio as tio
from monai.transforms import RandAffine, RandGaussianNoise, RandGaussianSmooth, RandRotate, RandZoom
from torchvision.transforms import Compose, RandomApply


class FractureTransforms:
    """Apply preprocessing and optional augmentation for CT patches."""

    def __init__(
        self,
        mode: str,
        input_size: int,
        input_dimension: int,
        image_augmentation_prob: float,
        augmentation_magnitude: int,
        num_sequential_transforms: int,
    ) -> None:
        self.mode = mode
        self.input_size = input_size
        self.input_dimension = input_dimension
        self.image_augmentation_prob = image_augmentation_prob
        self.augmentation_magnitude = augmentation_magnitude
        self.num_sequential_transforms = num_sequential_transforms

        max_magnitude = 10
        self.translation_range = (6 / max_magnitude) * augmentation_magnitude
        self.rotation_range = (1.05 / max_magnitude) * augmentation_magnitude
        self.zoom_range = (0.2 / max_magnitude) * augmentation_magnitude
        self.max_noise = 0.05
        self.max_smooth = 1.5

        self.preprocessing = Compose(
            [
                tio.Clamp(out_min=-100, out_max=1000),
                tio.RescaleIntensity(in_min_max=(-100, 1000), out_min_max=(0, 1)),
                tio.Resize((self.input_size, self.input_size, self.input_size)),
            ]
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rotation_x = RandRotate(range_x=self.rotation_range, padding_mode="zeros", prob=1)
        rotation_y = RandRotate(range_y=self.rotation_range, padding_mode="zeros", prob=1)
        rotation_z = RandRotate(range_z=self.rotation_range, padding_mode="zeros", prob=1)
        translate_x = RandAffine(translate_range=(self.translation_range, 0, 0), prob=1)
        translate_y = RandAffine(translate_range=(0, self.translation_range, 0), prob=1)
        translate_z = RandAffine(translate_range=(0, 0, self.translation_range), prob=1)
        zoom = RandZoom(min_zoom=(1 - self.zoom_range), max_zoom=(1 + self.zoom_range))
        smooth = RandGaussianSmooth(sigma_x=(0.1, self.max_smooth), sigma_y=(0.1, self.max_smooth), sigma_z=(0.1, self.max_smooth), prob=1)
        smooth_2d = RandGaussianSmooth(sigma_x=(0.1, self.max_smooth), sigma_y=(0.1, self.max_smooth), prob=1)
        noise = RandGaussianNoise(mean=0, std=self.max_noise, prob=1)
        flip = monai.transforms.RandFlip(prob=1)

        x = x[None, :, :, :]
        x = self.preprocessing(x)

        if self.input_dimension == 2:
            x = np.mean(x[:, 10:40, :], axis=1)
            x = np.broadcast_to(x, (3, x.shape[1], x.shape[2]))

        if self.mode == "training":
            transform_pool = [flip, rotation_x, rotation_y, translate_x, translate_y, zoom, smooth_2d, noise]
            if self.input_dimension == 3:
                transform_pool = [flip, rotation_x, rotation_y, rotation_z, translate_x, translate_y, translate_z, zoom, smooth, noise]
            sampled = random.sample(transform_pool, self.num_sequential_transforms)
            x = RandomApply(sampled, p=self.image_augmentation_prob)(x)

        return x
