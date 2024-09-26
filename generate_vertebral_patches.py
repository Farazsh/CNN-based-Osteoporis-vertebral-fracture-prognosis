import time

import SimpleITK
import numpy as np
import pandas as pd
import copy
import click
import json
import psutil
import os
import fnmatch
from pathlib import Path
from scipy import ndimage
from typing import List
from multiprocessing import Pool
from itertools import repeat
import matplotlib.pyplot as plt
from multiprocessing import pool
from scipy.ndimage import binary_dilation
import re

from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    Orientationd,
    SpatialCropd,
    Spacingd,
    SaveImaged
)


def transform_physical_point_to_continuous_index(image_path, pos_real_world):
    """
    Reads the metadata into a dummy image and transforms physical cordinates to image indices
    Its several times faster than reading the image data
    :param image_path:
    :param pos_real_world:
    :return:
    """
    reader = SimpleITK.ImageFileReader()
    reader.SetFileName(str(image_path))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    s = SimpleITK.GetImageFromArray(np.zeros((1, 1, 1)))  # dummy image
    s.SetSpacing(reader.GetSpacing())
    s.SetOrigin(reader.GetOrigin())
    s.SetDirection(reader.GetDirection())
    index = s.TransformPhysicalPointToContinuousIndex(pos_real_world)
    return np.array(index)


def calculate_cropsize_and_coordinates(img_path, patient_name, vertebra, df, crop_size, spacing):

    vertebra_row1 = df[(df.SubjectID == patient_name) & (df.Vertebra == vertebra)].to_numpy()[0].tolist()[3:6]
    center_coordinates1 = transform_physical_point_to_continuous_index(img_path, vertebra_row1)
    new_crop_size = [round(crop_size[i]/spacing[i]) for i in range(len(crop_size))]
    center_coordinates = [round(x) for x in center_coordinates1]
    return new_crop_size, center_coordinates


def crop_image(patient_name: str, vert: str, nii_path: str, df, output_dir: Path, plot_png: bool, crop_size):

    img_path = Path(rf"{nii_path}/{patient_name}.nii.gz")
    load_image = Compose([LoadImaged(keys="image", ensure_channel_first=True)])
    data = {"image": img_path}
    nii_image = load_image(data)
    spacing = nii_image['image_meta_dict']['pixdim'][1:4]

    new_crop_size, center_coordinates = calculate_cropsize_and_coordinates(img_path, patient_name,
                                                                           vert, df, crop_size, spacing)

    # define image transformations and apply to image to get vertebra crop
    crop = Compose([
        SpatialCropd(keys='image', roi_center=center_coordinates,
                     roi_size=new_crop_size),
        Orientationd(keys="image", axcodes='RAI'), Resized(keys='image', spatial_size=crop_size)])

    vertebra_crop = crop(nii_image)

    # save image as nifti patch
    save_nii = SaveImaged(keys='image', output_dir=output_dir,
                          separate_folder=False, output_postfix=vert,
                          resample=False, output_ext='.nii.gz')
    save_nii(vertebra_crop)

    if plot_png:
        # make crop bigger and save middle slice as png
        resize_crop = Resized(keys='image', spatial_size=[300, 300, 300])
        resized_slice = resize_crop(vertebra_crop)

        slice = resized_slice['image'][0, 150, :, :]
        slice = ndimage.rotate(slice, 90)
        plt.imsave(os.path.join(output_dir, 'png', f'{patient_name}_{vert}.png'), slice, cmap="gray")

    del vertebra_crop, nii_image


def print_sub_spacings():
    parent_dir = r"/data/MrOs"
    nii_path = os.path.join(parent_dir, r"MrOS US L1L2 Visit 1 CT Scans nii")
    dicom_path = os.path.join(parent_dir, r"MrOS US L1L2 Visit 1 CT Scans")
    load_image = Compose([LoadImaged(keys="image", ensure_channel_first=True)])
    subjects = {}
    for f in os.listdir(dicom_path):
        subjects[f] = list(os.listdir(os.path.join(dicom_path, f)))

    for f in subjects:
        count = 0
        while count != 10:
            sub = np.random.choice(subjects[f])
            try:
                file_path = Path(rf"{nii_path}/{sub}.nii.gz")
                nii_image = load_image({"image": file_path})
                print(sub, nii_image['image_meta_dict']['pixdim'][1:4])
                count += 1
            except:
                continue


def make_patches(patient_name: str, vert: str, nii_path: str, seg_path: str, df, df2, output_dir: Path, plot_png: bool, crop_size):
    try:
        crop_image(patient_name, vert, nii_path, seg_path, df, df2, output_dir, plot_png, crop_size)
    except:
        print("************* ERROR **********")
        print(patient_name)
        print("************* END ERROR **********")


def main():
    dfprime = pd.read_csv(r"MrOs_dataset/MrOs_Label_from_meta_v12_1_relabelled2.csv", index_col=None, delimiter=';')
    for col in ['X', 'Y', 'Z']:
        dfprime[col] = dfprime[col].apply(lambda x: x.replace(",", ".")).astype('float')
    sub_list = np.random.choice(dfprime['SubjectID'].unique().tolist(), 10)

    input_path = r"/data/MrOs/MrOS US L1L2 Visit 1 CT Scans nii"
    save_path = Path(r"/data/MrOs/MrOs_Vertebrae_patches/MrOS L1L2 202020 resized")
    png = True
    size = [60, 50, 40]

    # Patches using single processing
    for sub in sub_list:
        crop_image(sub, 'L1', input_path, dfprime, save_path, png, size)

    # Patches using multiprocessing
    pool = Pool()
    pool.starmap(make_patches,
                 zip(sub_list, repeat('L2'), repeat(input_path), repeat(dfprime), repeat(save_path), repeat(png),
                     repeat(size)))


if __name__ == '__main__':
    main()






















