<p align="center">
   <img src="https://github.com/BayraktarLab/cell2location/blob/master/docs/logo.svg?raw=True" width="200">
</p>

### Comprehensive mapping of tissue cell architecture via integrated single cell and spatial transcriptomics (cell2location model)

[![Stars](https://img.shields.io/github/stars/BayraktarLab/cell2location?logo=GitHub&color=yellow)](https://github.com/BayraktarLab/cell2location/stargazers)
![Build Status](https://github.com/BayraktarLab/cell2location/actions/workflows/test.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/cell2location/badge/?version=latest)](https://cell2location.readthedocs.io/en/stable/?badge=latest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BayraktarLab/cell2location/blob/master/docs/notebooks/cell2location_tutorial.ipynb)
[![Docker image on quay.io](https://img.shields.io/badge/container-quay.io/vitkl/cell2location-brightgreen "Docker image on quay.io")](https://quay.io/vitkl/cell2location)

Preprint: https://www.biorxiv.org/content/10.1101/2020.11.15.378125v1 

Cell2location is a principled Bayesian model that can resolve fine-grained cell types in spatial transcriptomic data and create comprehensive cellular maps of diverse tissues. Cell2location accounts for technical sources of variation and borrows statistical strength across locations, thereby enabling the integration of single cell and spatial transcriptomics with higher sensitivity and resolution than existing tools. This is achieved by estimating which combination of cell types in which cell abundance could have given the mRNA counts in the spatial data, while modelling technical effects (platform/technology effect, contaminating RNA, unexplained variance).

<p align="center">
   <img src="https://github.com/BayraktarLab/cell2location/blob/master/docs/images/Fig1_v2_white_bg.png?raw=True">
</p>
Overview of the spatial mapping approach and the workflow enabled by cell2location. From left to right: Single-cell RNA-seq and spatial transcriptomics profiles are generated from the same tissue (1). Cell2location takes scRNA-seq derived cell type reference signatures and spatial transcriptomics data as input (2, 3). The model then decomposes spatially resolved multi-cell RNA counts matrices into the reference signatures, thereby establishing a spatial mapping of cell types (4).    

## Usage and Tutorials

The tutorial covering the estimation of expresson signatures of reference cell types, spatial mapping with cell2location and the downstream analysis can be found here: https://cell2location.readthedocs.io/en/latest/

You can also try cell2location on [Google Colab](https://colab.research.google.com/github/BayraktarLab/cell2location/blob/master/docs/notebooks/cell2location_tutorial.ipynb) on a smaller data subset containing somatosensory cortex.

Please report bugs via https://github.com/BayraktarLab/cell2location/issues and ask any usage questions in https://github.com/BayraktarLab/cell2location/discussions.

Cell2location package is implemented in a general way (using https://pyro.ai/ and https://scvi-tools.org/) to support multiple related models - both for spatial mapping, estimating reference cell type signatures and downstream analysis.

## Installation

We suggest using a separate conda environment for installing cell2location.

Create conda environment and install `cell2location` package

```bash
conda create -y -n cell2loc_env python=3.9

conda activate cell2loc_env
pip install git+https://github.com/BayraktarLab/cell2location.git#egg=cell2location[tutorials]
```

Finally, to use this environment in jupyter notebook, add jupyter kernel for this environment:

```bash
conda activate cell2loc_env
python -m ipykernel install --user --name=cell2loc_env --display-name='Environment (cell2loc_env)'
```

If you do not have conda please install Miniconda first:

```bash
cd /path/to/software
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use prefix /path/to/software/miniconda3
```

Before installing cell2location and it's dependencies, it could be necessary to make sure that you are creating a fully isolated conda environment by telling python to NOT use user site for installing packages, ideally by adding this line to your `~/.bashrc` file , but this would also work during a terminal session:

```bash
export PYTHONNOUSERSITE="someletters"
```


## Documentation and API details

User documentation is availlable on https://cell2location.readthedocs.io/en/latest/. 

Cell2location architecture is designed to simplify extended versions of the model that account for additional technical and biologial information. We plan to provide a tutorial showing how to add new model classes but please get in touch if you would like to contribute or build on top our package.

## Acknowledgements 

We thank all paper authors for their contributions:
Vitalii Kleshchevnikov, Artem Shmatko, Emma Dann, Alexander Aivazidis, Hamish W King, Tong Li, Artem Lomakin, Veronika Kedlian, Mika Sarkin Jain, Jun Sung Park, Lauma Ramona, Liz Tuck, Anna Arutyunyan, Roser Vento-Tormo, Moritz Gerstung, Louisa James, Oliver Stegle, Omer Ali Bayraktar

We also thank Krzysztof Polanski, Luz Garcia Alonso, Carlos Talavera-Lopez, Ni Huang for feedback on the package, Martin Prete for dockerising cell2location and other software support.

## FAQ

See https://github.com/BayraktarLab/cell2location/discussions

## Future development and experimental features

We also provide an experimental numpyro translation of the model which has improved memory efficiency (allowing analysis of multiple Visium samples on Google Colab) and minor improvements in speed - https://github.com/vitkl/cell2location_numpyro. You can try it on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vitkl/cell2location_numpyro/blob/main/docs/notebooks/cell2location_short_demo_colab.ipynb) - however note that both numpyro itself and cell2location_numpyro are in very active development. 
