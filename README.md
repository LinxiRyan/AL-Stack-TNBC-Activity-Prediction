# AL-Stack-TNBC-Activity-Prediction

Reference implementation for the generation-augmented self-training and stacking framework for predicting molecular growth-inhibitory activity (pGI50) in MDA-MB-231 triple-negative breast cancer (TNBC) cells.

This repository provides study-specific code for data preprocessing, molecular representations, feature selection, stacking regression, iterative sample selection, REINVENT 4 data preparation and post-processing, and model interpretation. The implementation is modular and can be adapted to other datasets.

## Data Availability

The molecular activity data used in this study were obtained from publicly available resources, including [ChEMBL (release 35)](https://www.ebi.ac.uk/chembl/) and the [NCI anticancer screening database (via CellMiner)](https://dtp.cancer.gov/).

To support transparency and reproducibility, the study-specific datasets, source data annotations, computational codes, and supplementary materials requested during peer review are made available through this repository and its associated [GitHub Releases](https://github.com/LinxiRyan/AL-Stack-TNBC-Activity-Prediction/releases). These resources include processed molecular activity datasets, generated molecular data and associated pseudo-labels, the external validation dataset comprising 929 compounds with activity annotations, and supporting configuration files and analysis scripts.

Detailed descriptions of the available resources are provided in the repository documentation and corresponding release notes. Additional supporting materials and updates may be provided as necessary.

## Molecular Generation

Molecular generation was performed using [REINVENT 4](https://github.com/MolecularAI/REINVENT4), with separate generation tasks for the original training-data pGI50 intervals [5,6), [6,7), [7,8), and [8,9]. Study-specific data preparation and post-processing procedures are implemented in `src/`.

REINVENT 4 is maintained by its original developers and must be installed separately. Its source code and pretrained model weights are not redistributed here. Study-specific experimental settings and supporting configuration materials are documented in the repository or the associated release materials, as applicable.

## Repository Structure

The main source-code modules are organized as follows:

```text
.
├── README.md
├── requirements.txt
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

Additional datasets, experiment configurations, and supporting files may be provided directly in the repository or as downloadable assets in GitHub Releases. Their organization may be updated as further materials are made available.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

For RDKit, installation through conda-forge is recommended:

```bash
conda install -c conda-forge rdkit
```

REINVENT 4 must be installed separately when reproducing the molecular generation stage.

## Usage and Reproducibility

The scripts in `src/` implement individual stages of the computational workflow. Refer to the available datasets, configuration materials, and release notes for study-specific inputs and settings. Users may adapt file paths and computational configurations to their environments.

The released materials are intended to support methodological inspection and computational reproducibility. Full trained-model checkpoints and exhaustive intermediate training logs are not necessarily included. Model-assigned pseudo-labels and molecular docking results are computational outputs, not experimental validation of cellular activity or target binding.

## Citation

If you use this code or associated data, please cite:

*A Generation-Augmented Self-Training Framework for Anti-TNBC Activity Prediction with Imbalanced Data.* [Full citation to be added after publication.]
