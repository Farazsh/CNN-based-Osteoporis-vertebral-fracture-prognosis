# Osteoporotic Vertebral Fracture Prognosis from CT

A clean, research-oriented deep learning project for predicting long-term vertebral fracture risk from CT-derived vertebral patches.

## Why this project

Osteoporosis-related vertebral fractures are common and frequently under-detected. This repository provides an end-to-end pipeline to:

1. preprocess MrOS CT imaging data,
2. train 2D/3D CNN prognostic models,
3. evaluate classification and risk metrics,
4. run repeatable experiments with YAML-driven configuration.

## Dataset

This work is designed around the **MrOS** cohort (https://mrosonline.ucsf.edu/). The repository assumes access to:

- CT scans (DICOM),
- vertebral metadata/coordinates,
- split-aware label files for cross-validation.

> This repository does not redistribute dataset files.

## Project structure

```text
configs/
  default.yaml
  _legacy_config.py
src/osteo_fracture/
  config.py
  data/
    datamodule.py
    transforms.py
  models/
    backbones.py
  training/
    lightning_module.py
    losses.py
    metrics.py
    callbacks.py
  experiments/
    cross_validation.py
    hyperparameter_search.py
  utils/
    reproducibility.py
    logging.py
scripts/
  preprocess_dicom.py
  generate_patches.py
train.py
cli.py
```

## Architecture and training pipeline

- **Backbones**: fNet, ResNet variants, and SEResNeXt-compatible options.
- **Trainer**: PyTorch Lightning with checkpointing, early stopping, and optional SWA.
- **Metrics**: AUROC, AUPRC, Accuracy, Precision, Recall, F1, Specificity, plus risk-oriented C-index and sHR proxy.
- **Configuration**: YAML-first + dataclass parsing.

## Usage

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Train

```bash
python train.py
# or
python cli.py train --config configs/default.yaml
```

### 3) Preprocess raw data

```bash
python cli.py preprocess --dicom-root /path/to/dicom --output-root /path/to/nifti
```

### 4) Run cross-validation

```bash
python cli.py crossval --config configs/default.yaml --splits Split1 --splits Split2 --splits Split3 --splits Split4
```

### 5) Hyperparameter search

```bash
python cli.py hp-search --config configs/default.yaml --lr 1e-3 --lr 1e-4 --model fnet --model resnet18
```

## Example results (from associated study)

| Model         | AUROC ± SD | AUPRC ± SD | C-index | sHR |
|---------------|------------|------------|---------|-----|
| 2D fNet       | 80.5 ± 5.7 | 23.2 ± 14.9| 0.77    | 2.6 |
| 3D fNet       | 81.5 ± 4.7 | 23.1 ± 14.6| 0.78    | 2.5 |
| 2D ResNet18   | 80.7 ± 4.4 | 25.0 ± 13.0| 0.78    | 2.2 |
| 3D ResNet18   | 75.9 ± 1.1 | 13.8 ± 3.1 | 0.75    | 2.2 |

## Research-only disclaimer

This repository is provided for **research and educational use only**. It is **not** a medical device and must not be used for clinical decision-making without independent validation and regulatory approval.
