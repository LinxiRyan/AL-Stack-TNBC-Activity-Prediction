# AL-Stack-TNBC-Activity-Prediction

Reference implementation for the generation-augmented self-training and stacking framework for predicting molecular growth-inhibitory activity (pGI50) in MDA-MB-231 triple-negative breast cancer (TNBC) cells.

This repository provides study-specific code for data preprocessing, molecular representations, feature selection, stacking regression, iterative sample selection, REINVENT 4 data preparation and post-processing, and model interpretation. The implementation is modular and can be adapted to other datasets.

## Data Availability

The molecular activity data used in this study were obtained from the publicly available [ChEMBL database (release 35)](https://www.ebi.ac.uk/chembl/) and [NCI-60/DTP data (via CellMiner)](https://dtp.cancer.gov/). The study-specific processed datasets, including the modeling dataset, generated molecular candidates, and external validation set, are available as **separate ZIP archives** on the [GitHub Releases page](https://github.com/LinxiRyan/AL-Stack-TNBC-Activity-Prediction/releases). Raw source-database exports are not redistributed.

## Molecular Generation

Molecular generation used [REINVENT 4](https://github.com/MolecularAI/REINVENT4), with separate generation tasks for the original training-data pGI50 intervals [5,6), [6,7), [7,8), and [8,9]. The generation preparation and post-processing code is in `src/`.

The `configs/reinvent/` directory contains the study-specific REINVENT transfer-learning and mol2mol sampling configurations (including the prior model identifier, sampling settings, seed settings, and interval-specific augmentation factors). The REINVENT 4 source code, pretrained model weights, and fine-tuned checkpoints are not redistributed; obtain the software and applicable pretrained models from the official project.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── model_hyperparameters.yaml
│   └── reinvent/
└── src/
    ├── preprocessing.py
    ├── fingerprint.py
    ├── descriptors.py
    ├── feature_selection.py
    ├── stacking_model.py
    ├── sample_selection.py
    ├── active_learning.py
    ├── evaluation.py
    ├── interpretability.py
    ├── prepare_generation_data.py
    └── postprocess_generated.py
```

**Data files are distributed separately as ZIP archives via GitHub Releases.** After downloading and extracting them, place the CSV files in a local `data/` folder; this folder is not part of the tracked repository structure shown above.

`configs/model_hyperparameters.yaml` records the study-specific CatBoost, XGBoost, MLP, and Ridge settings, together with the hyperparameter-tuning procedure. The documented implementation uses the original train/validation/test protocol; any additional scaffold-split evaluation should be distinguished from that original split.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

For RDKit, conda-forge installation is recommended:

```bash
conda install -c conda-forge rdkit
```

REINVENT 4 must be installed separately when reproducing the generation stage.

## Usage and Reproducibility

The `src/` scripts implement individual stages of the workflow. Download the required ZIP archive(s) from the [GitHub Releases page](https://github.com/LinxiRyan/AL-Stack-TNBC-Activity-Prediction/releases), extract the CSV file(s), and place them in a local `data/` folder, or adjust input paths as needed. Use these tables and the study-specific configurations to inspect or reproduce the relevant stages. The generation and model configurations should be used with the software versions documented in `requirements.txt` and the official REINVENT 4 installation.

The released material is intended to support methodological inspection and computational reproduction. Full trained-model checkpoints and exhaustive intermediate training logs are not included. The provided pseudo-labels and docking predictions are computational outputs and do not constitute experimental activity or target-binding validation.

## Citation

If you use this code or associated data, please cite:

*A Generation-Augmented Self-Training Framework for Anti-TNBC Activity Prediction with Imbalanced Data.* [Full citation to be added after publication.]
