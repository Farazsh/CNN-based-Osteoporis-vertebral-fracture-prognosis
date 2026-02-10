"""Convert MrOS DICOM studies to NIfTI volumes."""

from __future__ import annotations

from pathlib import Path
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk


def uniform_spacing(volume: sitk.Image, new_spacing: list[float] | tuple[float, float, float] = (1, 1, 1)) -> sitk.Image:
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), sitk.sitkLinear, volume.GetOrigin(), new_spacing, volume.GetDirection(), 0, volume.GetPixelID())


def convert_series(dicom_dir: Path, output_path: Path) -> None:
    reader = sitk.ImageSeriesReader()
    series = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series:
        return
    names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series[0])
    reader.SetFileNames(names)
    image = reader.Execute()
    sitk.WriteImage(uniform_spacing(image), str(output_path))


def preprocess_site(site_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for subject in site_dir.iterdir():
        if not subject.is_dir():
            continue
        convert_series(subject, output_dir / f"{subject.name}.nii.gz")


def main(input_root: str, output_root: str, num_workers: int = 4) -> None:
    root = Path(input_root)
    out = Path(output_root)
    sites = [p for p in root.iterdir() if p.is_dir()]
    with Pool(processes=num_workers) as pool:
        pool.starmap(preprocess_site, [(site, out) for site in sites])


if __name__ == "__main__":
    main("/data/MrOs/MrOS US L1L2 Visit 1 CT Scans", "/data/MrOs/MrOS_US_L1L2_nifti")
