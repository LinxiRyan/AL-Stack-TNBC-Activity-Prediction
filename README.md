# AL-Stack-TNBC-Activity-Prediction

Reference implementation for an active learning and stacking-based
molecular activity prediction framework for TNBC cells.

This repository provides the computational implementation associated
with the study, including molecular data preprocessing, molecular
representation generation, feature selection, machine learning modeling,
active learning-based sample enrichment, molecular generation
post-processing, and model interpretation.

The released code is designed as a modular framework. Users can adapt
individual components according to their own molecular datasets and
computational requirements.

------------------------------------------------------------------------

Data Availability

The datasets used in this study were obtained from publicly available
databases.

ChEMBL

Bioactivity data were collected from the ChEMBL database:

https://www.ebi.ac.uk/chembl/

Users can retrieve molecular activity records through the ChEMBL
platform, including:

-   Molecular structures
-   Bioactivity measurements
-   Assay information
-   Target and cell-line related annotations

The downloaded ChEMBL records can be processed using the provided
scripts to generate molecular activity datasets suitable for model
development.

------------------------------------------------------------------------

NCI Anticancer Screening Data

Additional molecular activity data were obtained from the NCI anticancer
screening database:

https://dtp.cancer.gov/

Users can retrieve:

-   Compound identifiers
-   Molecular structures
-   Cell-line-specific activity measurements

The corresponding downloaded records can be processed using the provided
preprocessing workflow.

The raw datasets are not redistributed in this repository. Users should
obtain the original data directly from the corresponding public
databases.

------------------------------------------------------------------------

Molecular Generation

Molecular generation was performed using REINVENT4:

https://github.com/MolecularAI/REINVENT4

REINVENT4 is an AI-driven molecular design framework supporting transfer
learning, reinforcement learning, and molecular optimization.

In this study, activity-associated molecular subsets were prepared and
used with REINVENT4-based molecular generation workflows. The released
repository only contains study-specific preparation and post-processing
procedures.

The following materials are not included:

-   REINVENT4 source code
-   Pretrained models
-   Fine-tuned checkpoints
-   Experiment-specific configuration files
-   Sampling parameters

Users should install and configure REINVENT4 following the official
documentation and specify appropriate generation settings according to
their own requirements.

------------------------------------------------------------------------

Repository Structure

    .
    ├── README.md
    ├── requirements.txt
    │
    └── src
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

------------------------------------------------------------------------

Installation

Install the required Python dependencies:

    pip install -r requirements.txt

For RDKit, installation through conda-forge is recommended:

    conda install -c conda-forge rdkit

The code was developed and tested in a Python-based scientific computing
environment. Users may need to adjust package versions according to
their operating system and hardware configuration.

------------------------------------------------------------------------

Usage

The repository provides modular implementations for different stages of
the computational workflow.

A typical usage scenario includes:

1.  Downloading molecular activity data from ChEMBL and NCI databases.
2.  Preparing and standardizing molecular structures.
3.  Generating molecular fingerprints and descriptors.
4.  Performing feature selection.
5.  Training machine learning models.
6.  Applying active learning strategies for molecular selection.
7.  Processing generated molecules obtained from molecular generation
    workflows.
8.  Performing model evaluation and interpretation.

The released implementation is intended to provide the methodological
framework. Users should define dataset-specific parameters, model
configurations, and computational settings according to their own
applications.

------------------------------------------------------------------------

Reproducibility Note

Due to confidentiality restrictions, some experiment-specific materials
are not distributed, including:

-   Raw processed datasets
-   Trained model checkpoints
-   Internal optimization configurations
-   Experiment-specific hyperparameters
-   Molecular generation configuration files

The released repository provides the computational workflow and can be
adapted using publicly available molecular activity datasets.

------------------------------------------------------------------------

Citation

If you use this code, please cite:

[Publication information will be added after publication]
