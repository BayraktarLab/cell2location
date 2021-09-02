<p align="center">
   <img src="https://github.com/BayraktarLab/cell2location/blob/master/docs/logo.svg?raw=True" width="200">
</p>

### Comprehensive mapping of tissue cell architecture via integrated single cell and spatial transcriptomics (cell2location model)

[![Docker Repository on Quay](https://quay.io/repository/vitkl/cell2location/status "Docker Repository on Quay")](https://quay.io/repository/vitkl/cell2location)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BayraktarLab/cell2location/blob/master/docs/notebooks/cell2location.ipynb)

Preprint: https://www.biorxiv.org/content/10.1101/2020.11.15.378125v1 

Cell2location is a principled Bayesian model that can resolve fine-grained cell types in spatial transcriptomic data and create comprehensive cellular maps of diverse tissues. Cell2location accounts for technical sources of variation and borrows statistical strength across locations, thereby enabling the integration of single cell and spatial transcriptomics with higher sensitivity and resolution than existing tools. This is achieved by estimating which combination of cell types in which cell abundance could have given the mRNA counts in the spatial data, while modelling technical effects (platform/technology effect, contaminating RNA, unexplained variance).

<p align="center">
   <img src="https://github.com/BayraktarLab/cell2location/blob/master/docs/images/Fig1_v2_white_bg.png?raw=True">
</p>
Overview of the spatial mapping approach and the workflow enabled by cell2location. From left to right: Single-cell RNA-seq and spatial transcriptomics profiles are generated from the same tissue (1). Cell2location takes scRNA-seq derived cell type reference signatures and spatial transcriptomics data as input (2, 3). The model then decomposes spatially resolved multi-cell RNA counts matrices into the reference signatures, thereby establishing a spatial mapping of cell types (4).    

## Usage and Tutorials

Tutorials covering the estimation of expresson signatures of reference cell types (1/3), spatial mapping with cell2location (2/3) and the downstream analysis (3/3) can be found here: https://cell2location.readthedocs.io/en/latest/

There are 2 ways to install and use our package: setup your [own conda environment](https://github.com/BayraktarLab/cell2location#installation-of-dependecies-and-configuring-environment) or use the [singularity](https://github.com/BayraktarLab/cell2location#using-singularity-image) and [docker](https://github.com/BayraktarLab/cell2location#using-docker-image) images (recommended). See below for details.

You can also try cell2location on [Google Colab](https://colab.research.google.com/github/BayraktarLab/cell2location/blob/master/docs/notebooks/cell2location.ipynb) on a smaller data subset containing somatosensory cortex.

Please report buga via https://github.com/BayraktarLab/cell2location/issues and ask any usage questions in https://github.com/BayraktarLab/cell2location/discussions.

We also provide an experimental numpyro translation of the model which has improved memory efficiency (allowing analysis of multiple Visium samples on Google Colab) and minor improvements in speed - https://github.com/vitkl/cell2location_numpyro. You can try it on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vitkl/cell2location_numpyro/blob/main/docs/notebooks/cell2location_short_demo_colab.ipynb) - however note that both numpyro itself and cell2location_numpyro are in very active development. 

Cell2location package is implemented in a general way to support multiple related models - both for spatial mapping and estimating signatures of cell types (tutorials use default models - no need to change):
1. `LocationModelLinearDependentWMultiExperiment` - main model for estimating cell abundance by decomposing spatial data into reference expression signatures of cell types.
2. `LocationModelWTA` - same as in #1 but adapted to work with Nanostring WTA data
3. Similified versions of model #1 that lack particular features of the full model, accessible from `cell2location.models.simplified`

Models for estimating reference expression signatures of cell types from scRNA data:
1. `RegressionGeneBackgroundCoverageTorch` - estimating expression signatures of cell types, accounting for variable sequencing depth between batches (e.g. 10X reaction) and additive background (contaminating RNA).
2. `RegressionGeneBackgroundCoverageGeneTechnologyTorch` - similar to #1 but additionally accounts for multiplicative platform effect between scRNA technologies.

Additionally we provide 2 models for downstream analysis of cell abundance estimates, accessible from `cell2location.models.downstream`:
1. `CoLocatedGroupsSklearnNMF` - identifying groups of cell types with similar locations using NMF (wrapper around sklearn NMF). See tutorial #3 for usage.
2. `ArchetypalAnalysis` - identifying smoothly varying and mutually exclusive tissue zones with Archetypa Analysis.

## Installation

Prior to installing cell2location package you need to install miniconda and create a conda environment containing pymc3 and theano ready for use on GPU. Follow the steps below:

If you do not have conda please install Miniconda first:

```bash
cd /path/to/software
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use prefix /path/to/software/miniconda3
```

Before installing cell2location and it's dependencies, make sure that you are creating a fully isolated conda environment by telling python to NOT use user site for installing packages, ideally by adding this line to your `~/.bashrc` file , but this would also work during a terminal session:
```bash
export PYTHONNOUSERSITE="someletters"
```

#### 1. Create conda environment manually

Create conda environment with the required packages pymc3 and scanpy:

```bash
conda create -n cellpymc python=3.7 numpy pandas jupyter leidenalg python-igraph scanpy \
louvain hyperopt loompy cmake nose tornado dill ipython bbknn seaborn matplotlib request \
mkl-service pygpu --channel bioconda --channel conda-forge
```

#### 2. Install `cell2location` package

```bash
conda activate cellpymc
pip install git+https://github.com/BayraktarLab/cell2location.git
```

Finally, to use this environment in jupyter notebook, add jupyter kernel for this environment:

```bash
conda activate cellpymc
python -m ipykernel install --user --name=cellpymc --display-name='Environment (cellpymc)'
```

## Documentation and API details

User documentation is availlable on https://cell2location.readthedocs.io/en/latest/. 

The architecture of the package is briefly described [here](https://github.com/BayraktarLab/cell2location/blob/master/cell2location/models/README.md). Cell2location architecture is designed to simplify extended versions of the model that account for additional technical and biologial information. We plan to provide a tutorial showing how to add new model classes but please get in touch if you would like to contribute or build on top our package.

We also provide an experimental numpyro translation of the model. Note that the pyro translation of cell2location in this repo does not work.

## Acknowledgements 

We thank all paper authors for their contributions:
Vitalii Kleshchevnikov, Artem Shmatko, Emma Dann, Alexander Aivazidis, Hamish W King, Tong Li, Artem Lomakin, Veronika Kedlian, Mika Sarkin Jain, Jun Sung Park, Lauma Ramona, Liz Tuck, Anna Arutyunyan, Roser Vento-Tormo, Moritz Gerstung, Louisa James, Oliver Stegle, Omer Ali Bayraktar

We also thank Krzysztof Polanski, Luz Garcia Alonso, Carlos Talavera-Lopez, Ni Huang for feedback on the package, Martin Prete for dockerising cell2location and other software support.

## FAQ

See https://github.com/BayraktarLab/cell2location/discussions
