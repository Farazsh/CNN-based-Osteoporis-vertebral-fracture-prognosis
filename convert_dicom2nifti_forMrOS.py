import os
import SimpleITK as sitk
import numpy as np
from os.path import join
from pathlib import Path
from multiprocessing import Pool
from itertools import repeat


def get_slice_coordinates(dicom_path, uid):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path, seriesID=uid)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    slice_locations = []

    for f in dicom_names:
        freader = sitk.ImageFileReader()
        freader.SetFileName(f)
        freader.LoadPrivateTagsOn()
        freader.ReadImageInformation()
        slice_locations.append(float(freader.GetMetaData('0020|1041')))

    return slice_locations, image


# Deprecated
def create_affine_matrix(image):
    # Deprecated
    affine = np.eye(4)
    for i in range(3):
        affine[i, i] = image.GetSpacing()[i]
    for i in range(3):
        affine[i, 3] = image.GetOrigin()[i]
    return affine


# Deprecated
def single_sub_dicom2nifti(parent_dir, subject_name, site, save_path, extension=''):
    if extension != '':
        dicom_dir = join(parent_dir, subject_name, extension)
    else:
        dicom_dir = join(parent_dir, subject_name)
    if len(os.listdir(dicom_dir)) < 5:
        print(subject_name, "Number of files:", len(os.listdir(dicom_dir)))
        return
    reader = sitk.ImageSeriesReader()
    series_UIDs = reader.GetGDCMSeriesIDs(dicom_dir)

    if len(series_UIDs) == 0:
        print("No dicom files for subject", subject_name)

    elif len(series_UIDs) == 1:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, join(save_path, f"{subject_name}{extension.split('_')[-1]}.nii.gz"))
        print("Processed successfully", subject_name, "dicom_series:", len(series_UIDs))

    else:
        print("Multiple dicom series, use another method")


def check_for_gaps_or_overlaps(slice1_ids, slice2_ids, subject_name):
    """
    Checks whether there is irregularity in two dicom series in terms of spacing, gaps and overlapping
    :param slice1_ids:
    :param slice2_ids:
    :param subject_name:
    :return:
    """
    spacing1 = np.diff(slice1_ids)
    if not np.all(np.isclose(spacing1, spacing1[0])):
        print("Irregular spacing in slice1", subject_name)

    spacing2 = np.diff(slice2_ids)
    if not np.all(np.isclose(spacing2, spacing2[0])):
        print("Irregular spacing in slice2", subject_name)

    if np.abs(spacing1[0]) != spacing2[0]:
        print("Spacing in both slices do not match", subject_name)

    if not np.isclose(slice2_ids[0] - slice1_ids[-1], np.abs(spacing1[0])):
        print("Possible gap or overlap", subject_name)


def uniform_spacing(volume, interpolator=sitk.sitkLinear, new_spacing=[1, 1, 1]):
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())


def process_single_dicom_series(dicom_dir, subject_name, save_path):
    """
    Coverts all files of one single dicom series to nifti
    :param dicom_dir:
    :param subject_name:
    :param save_path:
    :return:
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image = uniform_spacing(image)
    sitk.WriteImage(image, join(save_path, f"{subject_name}.nii.gz"))
    print("Processed successfully", subject_name, "num. of dicom series: 1")


def combine_multiple_dicoms(parent_dir, subject_name, site_name, save_path):
    """
    Checks the number of folders, dicom series present in the folder and accordingly process the dicom to niftii
    :param site_name:
    :param parent_dir:
    :param subject_name:
    :param save_path:
    :return:
    """
    dicom_dir = join(parent_dir, site_name, subject_name)

    # For BI and PI site
    if len(os.listdir(dicom_dir)) == 2 and np.all([os.path.isdir(join(dicom_dir, f)) for f in os.listdir(dicom_dir)]):
        vertebra_suffix = {'a': 'L1', 'b': 'L2'}
        for f in os.listdir(dicom_dir):
            process_single_dicom_series(join(dicom_dir, f), subject_name+'_'+vertebra_suffix[f.split("_")[-1]], save_path)
        return

    if len(os.listdir(dicom_dir)) < 5:
        print("Less than 5 files in folder for subject:", subject_name)

    reader = sitk.ImageSeriesReader()
    series_UIDs = reader.GetGDCMSeriesIDs(dicom_dir)

    if len(series_UIDs) == 0:
        print("No dicom files for subject", subject_name)

    elif len(series_UIDs) == 1:
        process_single_dicom_series(dicom_dir, subject_name, save_path)

    # For PI site
    elif len(series_UIDs) == 2:
        s_loc1, img1 = get_slice_coordinates(dicom_dir, series_UIDs[0])
        img_arr1 = sitk.GetArrayFromImage(img1)
        s_loc2, img2 = get_slice_coordinates(dicom_dir, series_UIDs[1])
        img_arr2 = sitk.GetArrayFromImage(img2)

        if np.all(s_loc1 < s_loc2):
            check_for_gaps_or_overlaps(s_loc1, s_loc2, subject_name)
            stacked_array = np.concatenate([img_arr1, img_arr2], axis=0)
            seed_image = img1
        elif np.all(s_loc1 > s_loc2):
            check_for_gaps_or_overlaps(s_loc2, s_loc1, subject_name)
            stacked_array = np.concatenate([img_arr2, img_arr1], axis=0)
            seed_image = img2
        else:
            print('Error in slice locations for subject: ', subject_name)
            return

        processed_image = sitk.GetImageFromArray(stacked_array)
        processed_image.SetOrigin(seed_image.GetOrigin())
        processed_image.SetSpacing(seed_image.GetSpacing())
        processed_image.SetDirection(seed_image.GetDirection())
        processed_image = uniform_spacing(processed_image)
        sitk.WriteImage(processed_image, join(save_path, f"{subject_name}.nii.gz"))
        print("Processed successfully", subject_name, "dicom_series:", len(series_UIDs))

    else:
        print("More than 2 dicom series for subject", subject_name)


def convert_to_nii(parent_dir, subject_name, site_name, save_path):
    try:
        combine_multiple_dicoms(parent_dir, subject_name, site_name, save_path)
        return None
    except:
        return subject_name


def main():
    base_dir = r"/data/MrOs/MrOS US L1L2 Visit 1 CT Scans"
    save_dir = r"/data/MrOs/MrOS US L1L2 Visit 1 CT Scans nii 111 spacing"
    subjects = []
    sites = []
    for site in os.listdir(base_dir):
        if site=='BI':
            continue
        parent_dir = join(base_dir, site)
        subjects.extend(os.listdir(parent_dir))
        sites.extend([site]*len(os.listdir(parent_dir)))
    subjects = subjects

    pool = Pool()
    failed_subjects = pool.starmap(convert_to_nii, zip(repeat(base_dir), subjects, sites, repeat(save_dir)))
    failed_subjects = [x for x in failed_subjects if x]
    print(failed_subjects)


if __name__ == '__main__':
    main()
