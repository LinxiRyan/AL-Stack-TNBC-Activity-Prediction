# AL-Stack-TNBC-Activity-Prediction

Reference implementation for the generation-augmented self-training and stacking framework for predicting molecular growth-inhibitory activity (pGI50) in MDA-MB-231 triple-negative breast cancer (TNBC) cells.

This repository provides study-specific code for data preprocessing, molecular representations, feature selection, stacking regression, iterative sample selection, REINVENT 4 data preparation and post-processing, and model interpretation. The implementation is modular and can be adapted to other datasets.

## Data Availability

The original activity records were obtained from publicly available resources:

- **ChEMBL (release 35):** https://www.ebi.ac.uk/chembl/
- **NCI-60 / DTP (via CellMiner):** https://dtp.cancer.gov/

The raw source-database exports are not redistributed. To make the analyses in the manuscript auditable, the `data/` directory contains the following **study-specific processed tables**:

| File | Contents |
| --- | --- |
| `data/processed_dataset.csv` | Processed modeling compounds (n = 38,681), including canonical SMILES, pGI50, source database/record identifier, and original train/validation/test split assignment. |
| `data/generated_candidate_pool.csv` | Filtered REINVENT 4 candidate molecules, source activity bin, selection status/round, and model-assigned pseudo-pGI50 for selected candidates. A missing pseudo-label means that no pseudo-label was assigned to that candidate. |
| `data/external_validation_929.csv` | External categorical validation compounds (n = 929; 189 active and 740 inactive), including SMILES, reference class label, source/reference and available label provenance. |

The external active/inactive annotations originate from the cited source records; they are **not** assumed to have been assigned using one uniform experimental GI50 cutoff. The `pGI50` values assigned to generated molecules are **model-derived pseudo-labels, not experimental measurements**.

These processed tables document the study-specific preprocessing, split and external annotations that cannot be reconstructed unambiguously from the public databases alone.

## Molecular Generation

Molecular generation used [REINVENT 4](https://github.com/MolecularAI/REINVENT4), with separate generation tasks for the original training-data pGI50 intervals [5,6), [6,7), [7,8), and [8,9]. The generation preparation and post-processing code is in `src/`.

The `configs/reinvent/` directory contains the study-specific REINVENT transfer-learning and mol2mol sampling configurations (including the prior model identifier, sampling settings, seed settings and interval-specific augmentation factors). The REINVENT 4 source code, pretrained model weights, and fine-tuned checkpoints are not redistributed; obtain the software and applicable pretrained models from the official project.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── processed_dataset.csv
│   ├── generated_candidate_pool.csv
│   └── external_validation_929.csv
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

`configs/model_hyperparameters.yaml` records the study-specific CatBoost, XGBoost, MLP and Ridge settings, together with the hyperparameter-tuning procedure. The documented implementation uses the original train/validation/test protocol; any additional scaffold-split evaluation should be distinguished from that original split.

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

The `src/` scripts implement individual stages of the workflow. Use the processed tables and study-specific configurations to inspect or reproduce the relevant stages. The generation and model configurations should be used with the software versions documented in `requirements.txt` and the official REINVENT 4 installation.

The released material is intended to support methodological inspection and computational reproduction. Full trained-model checkpoints and exhaustive intermediate training logs are not included. The provided pseudo-labels and docking predictions are computational outputs and do not constitute experimental activity or target-binding validation.

## Citation

If you use this code or associated data, please cite:

*A Generation-Augmented Self-Training Framework for Anti-TNBC Activity Prediction with Imbalanced Data.* [Full citation to be added after publication.]
