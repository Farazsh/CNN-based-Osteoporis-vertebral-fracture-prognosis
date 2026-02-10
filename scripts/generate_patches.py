"""Generate vertebral patches from preprocessed NIfTI scans."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from monai.transforms import Compose, LoadImaged, Orientationd, Resized, SaveImaged, SpatialCropd


def transform_point_to_index(image_path: Path, world_xyz: tuple[float, float, float]) -> np.ndarray:
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(image_path))
    reader.ReadImageInformation()
    probe = sitk.GetImageFromArray(np.zeros((1, 1, 1)))
    probe.SetSpacing(reader.GetSpacing())
    probe.SetOrigin(reader.GetOrigin())
    probe.SetDirection(reader.GetDirection())
    return np.array(probe.TransformPhysicalPointToContinuousIndex(world_xyz))


def crop_vertebra(image_path: Path, center_xyz: tuple[float, float, float], patch_size: tuple[int, int, int], out_dir: Path, postfix: str) -> None:
    load_image = Compose([LoadImaged(keys="image", ensure_channel_first=True)])
    image_dict = load_image({"image": image_path})
    spacing = image_dict["image_meta_dict"]["pixdim"][1:4]
    center = [round(v) for v in transform_point_to_index(image_path, center_xyz)]
    roi_size = [round(patch_size[i] / spacing[i]) for i in range(3)]

    crop = Compose(
        [
            SpatialCropd(keys="image", roi_center=center, roi_size=roi_size),
            Orientationd(keys="image", axcodes="RAI"),
            Resized(keys="image", spatial_size=patch_size),
        ]
    )
    out = crop(image_dict)
    SaveImaged(keys="image", output_dir=str(out_dir), separate_folder=False, output_postfix=postfix, output_ext=".nii.gz", resample=False)(out)


def main(labels_csv: str, nifti_dir: str, output_dir: str) -> None:
    df = pd.read_csv(labels_csv, sep=";")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        image_path = Path(nifti_dir) / f"{row['SubjectID']}.nii.gz"
        center_xyz = (float(str(row["X"]).replace(",", ".")), float(str(row["Y"]).replace(",", ".")), float(str(row["Z"]).replace(",", ".")))
        crop_vertebra(image_path, center_xyz, (60, 50, 40), out, str(row["Vertebra"]))


if __name__ == "__main__":
    main(
        "MrOs_dataset/MrOs_Label_from_meta_v12_1_relabelled2.csv",
        "/data/MrOs/MrOS US L1L2 Visit 1 CT Scans nii",
        "/data/MrOs/MrOs_Vertebrae_patches",
    )
