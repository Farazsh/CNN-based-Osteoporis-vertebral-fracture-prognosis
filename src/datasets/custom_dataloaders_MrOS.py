import os
import random
import pandas as pd
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Final, List, Optional, Hashable, Dict, Tuple
import torch
import torchvision.transforms
from pandas import DataFrame, read_csv, Series
from torch.utils.data import DataLoader
import torchio as tio
import pytorch_lightning as pl
from torchvision.transforms import Compose, RandomApply
from src.io_tools import PROJECT_ROOT_DIR
import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import Transform, EnsureChannelFirst
from monai.transforms import RandRotate, RandZoom, RandShiftIntensity, RandGaussianNoise, RandGaussianSmooth, RandAffine
import numpy as np


class DataModuleMrOS(pl.LightningDataModule, ABC):

    def __init__(self, config):
        super().__init__()
        self._config = config.datamodule
        self._img_dir: Final[Path] = PROJECT_ROOT_DIR.joinpath(self._config.img_dir)
        self._labels_csv: Final[Path] = PROJECT_ROOT_DIR.joinpath(self._config.labels_file)
        self._train_set = None
        self._val_set = None
        self._test_set = None
        self._label_column_name = self._config.label
        self._censoring_label = self._config.censor_label  # regression
        self.ID_column_name = self._config.ID_label
        self.inference_only = self._config.Inference_only
        self.train_as_test = False
        self.validation_as_test = False
        self.test_as_validation = config.datamodule.test_as_validation

    def swap_dataset_labels(self, df, label_a, label_b) -> DataFrame:
        df[self._config.split_name] = df[self._config.split_name].replace({label_a: label_b, label_b: label_a})
        return df

    def _load_labels_df(self) -> DataFrame:
        df = pd.read_csv(self._labels_csv, sep=';')
        # df = df[df[self._label_column_name] == df[self._label_column_name]]  # remove NA values
        if self.train_as_test:
            self.swap_dataset_labels(df, 'Training', 'Testing')
        elif self.validation_as_test:
            self.swap_dataset_labels(df, 'Validation', 'Testing')
        elif self.test_as_validation:
            self.swap_dataset_labels(df, 'Testing', 'Validation')

        # For plotting only MIUA paper sample subjects
        # df = df[df.SubjectID.isin(['BI0370', 'MN1964', 'BI0083', 'PO6593', 'PA3225', 'PI4922', 'PO6588', 'SD8583', 'SD8059',
        #                            'PO7294', 'MN2094', 'MN1828', 'PO6588', 'BI0435', 'PO6588', 'SD8024', 'PA3232'])]

        # For plotting only BMI
        # df = df[(df.BMI>35) & (df.BMI<39)]
        # for i in range(1, 4):
        #     df[f'Split{i}'] = 'Testing'
        # df.sort_values(['SubjectID', 'Vertebra', 'AGE'], inplace=True)
        # df = df[df[self._config.label]==1]  # for plotting sagittal views

        # df[self._censoring_label] = df[self._censoring_label].astype(int)  # regression
        # df[self._label_column_name] = df[self._label_column_name].astype(int)
        return df

    def _build_dataset_from_subjectlist_and_apply_transformations(self, train_subjects, val_subjects, test_subjects):
        self._train_set = ImageDataset(image_files=train_subjects[0], labels=train_subjects[1],
                                       transform=ApplyTransformations('training', self._config))
        self._val_set = ImageDataset(image_files=val_subjects[0], labels=val_subjects[1],
                                     transform=ApplyTransformations('validation', self._config))
        self._test_set = ImageDataset(image_files=test_subjects[0], labels=test_subjects[1],
                                      transform=ApplyTransformations('testing', self._config))

    def get_class_ratio(self) -> float:
        labels_df: Final[DataFrame] = self._load_labels_df()
        df_slice = labels_df[labels_df[self._config.split_name] == 'Training']
        return len(df_slice[df_slice[self._config.label] == 0]) / len(df_slice[df_slice[self._config.label] == 1])

    def prepare_data(self) -> None:
        labels_df = self._load_labels_df()

        train_image_label_list = self.build_list_of_images(labels_df, "Training")
        validation_image_label_list = self.build_list_of_images(labels_df, "Validation")
        test_image_label_list = self.build_list_of_images(labels_df, "Testing")

        self._build_dataset_from_subjectlist_and_apply_transformations(train_image_label_list,
                                                                       validation_image_label_list,
                                                                       test_image_label_list)

    def build_list_of_images(self, labels_df: DataFrame, dataset_name: str) -> List:
        df_slice = labels_df[labels_df[self._config.split_name] == dataset_name]
        filenames = df_slice.Filename.tolist()
        image_list = [str(image.name) for image in self._img_dir.iterdir()]
        missing_images = list(set(filenames) - set(image_list))
        if len(missing_images) > 0:
            print(f"Following images are missing from {dataset_name} dataset:", missing_images)
        image_list = list(set(image_list).intersection(set(filenames)))

        if self.inference_only:
            label_list = [(self.get_label_by_filename(df_slice, label),
                           self.map_id_by_filename(df_slice, label)) for label in image_list]
        else:
            label_list = [self.get_label_by_filename(df_slice, label) for label in image_list]
        # label_list = [(self.get_label_by_filename(df_slice, label),  # Regression
        #                self.get_censoring_by_filename(df_slice, label)) for label in image_list]

        image_list = [str(Path.joinpath(self._img_dir, img)) for img in image_list]
        return [image_list, label_list]

    def get_label_by_filename(self, labels_df: DataFrame, filename: str) -> int:
        return labels_df.loc[labels_df['Filename'] == filename][self._label_column_name].values[0]

    def get_censoring_by_filename(self, labels_df: DataFrame, filename: str) -> int:
        return int(labels_df.loc[labels_df['Filename'] == filename][self._censoring_label].values)

    def map_id_by_filename(self, labels_df: DataFrame, filename: str) -> int:
        return int(labels_df.loc[labels_df['Filename'] == filename][self.ID_column_name].values)

    def get_dataloader(self, dataset, to_shuffle=False):
        return DataLoader(dataset=dataset,
                          batch_size=self._config.batch_size,
                          num_workers=self._config.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          shuffle=to_shuffle)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self._train_set, True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self._val_set)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self._test_set)


class ApplyTransformations(object):
    def __init__(self, mode, config):
        self._config = config
        self._mode = mode
        self.magnitude = self._config.augmentation_magnitude
        self.num_sequential_transforms = self._config.num_sequential_transforms
        self.img_augmentation_prob = self._config.image_augmentation_prob
        self.input_size = self._config.input_size
        self.input_dimension = self._config.input_dimension

        self._max_value = 1000
        self._min_value = -100
        self.max_magnitude = 10

        self.max_translation_pixels = 6
        self.max_rotation_radians = 1.05
        self.max_zoom = 0.2
        self.max_noise = 0.05
        self.max_smooth = 1.5
        self.max_shift = 0.3

        self.translation_range = (self.max_translation_pixels / self.max_magnitude) * self.magnitude
        self.rotation_range = (self.max_rotation_radians / self.max_magnitude) * self.magnitude
        self.zoom_range = (self.max_zoom / self.max_magnitude) * self.magnitude
        self.noise_range = (self.max_noise / self.max_magnitude) * self.magnitude
        self.smooth_range = (self.max_smooth / self.max_magnitude) * self.magnitude
        self.shift_range = (self.max_shift / self.max_magnitude) * self.magnitude

        self.preprocessing_transforms = Compose([
            # tio.Resample((47, 47, 47)),
            # tio.CropOrPad((self.input_size, self.input_size, self.input_size)),
            tio.Clamp(out_min=-100, out_max=1000),
            tio.RescaleIntensity(in_min_max=(-100, 1000), out_min_max=(0, 1)),
            tio.Resize((self.input_size, self.input_size, self.input_size)),
        ])

        # self.resize_2d = torchvision.transforms.Resize((47, 47))

    def __call__(self, x):
        rotation_x = RandRotate(range_x=self.rotation_range, padding_mode='zeros', prob=1)
        rotation_y = RandRotate(range_y=self.rotation_range, padding_mode='zeros', prob=1)
        rotation_z = RandRotate(range_z=self.rotation_range, padding_mode='zeros', prob=1)

        translate_x = RandAffine(translate_range=(self.translation_range, 0, 0), prob=1)
        translate_y = RandAffine(translate_range=(0, self.translation_range, 0), prob=1)
        translate_z = RandAffine(translate_range=(0, 0, self.translation_range), prob=1)

        zoom = RandZoom(min_zoom=(1 - self.zoom_range), max_zoom=(1 + self.zoom_range))
        shift_intensity = RandShiftIntensity(offsets=self.shift_range, prob=1)
        noise = RandGaussianNoise(mean=0, std=self.max_noise, prob=1)
        smooth = RandGaussianSmooth(sigma_x=(0.1, self.max_smooth), sigma_y=(0.1, self.max_smooth),
                                    sigma_z=(0.1, self.max_smooth), prob=1)
        smooth_2d = RandGaussianSmooth(sigma_x=(0.1, self.max_smooth), sigma_y=(0.1, self.max_smooth), prob=1)
        flip = monai.transforms.RandFlip(prob=1)

        # all_transforms = [rotation_x, rotation_y, rotation_z, translate_x, translate_y, translate_z, zoom, smooth]
        # sampled_transforms = random.sample(all_transforms, self.num_sequential_transforms)

        sampled_transforms = random.sample([flip, rotation_x, rotation_y, rotation_z,
                                            translate_x, translate_y, translate_z, zoom, smooth, noise],
                                           self._config.num_sequential_transforms)

        sampled_transforms_2d = random.sample([flip, rotation_x, rotation_y, translate_x,
                                               translate_y, zoom, smooth_2d, noise],
                                              self._config.num_sequential_transforms)
        training_transforms = RandomApply(sampled_transforms, p=self._config.image_augmentation_prob)
        training_transforms_2d = RandomApply(sampled_transforms_2d, p=self._config.image_augmentation_prob)
        x = x[None, :, :, :]  # monai transforms need chanel first 3d

        # when using 2d image with 3 channels and channel first
        # x = x[:, :, 0:3]  # remove alpha channel, doesn't affect other values
        # x = x.transpose((2, 0, 1))  # Makes channel last
        # x = self.resize_2d(torch.tensor(x))
        # when using 2d image with 1 channel
        # x = np.expand_dims(x, 0)
        # x = np.broadcast_to(x, (3, x.shape[1], x.shape[2]))  # monai transforms need chanel first 2d
        # print(x.shape)
        x = self.preprocessing_transforms(x)  # always apply preprocessing e.g. clamp and rescale for 3d

        if self.input_dimension == 2:
            x = np.mean(x[:, 10: 40, :], axis=1)
            # x = np.expand_dims(x, 0)
            x = np.broadcast_to(x, (3, x.shape[1], x.shape[2]))
            # x = x[:, 25, :]
        # x = x[:, 25, :]
        # x = np.mean(x, axis=1)
        if self._mode == 'training':  # only to rand augment on the training images
            if self.input_dimension == 2:
                x = training_transforms_2d(x)
            else:
                x = training_transforms(x)

        return x
