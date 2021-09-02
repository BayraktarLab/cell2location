<p align="center">
   <img src="https://github.com/BayraktarLab/cell2location/blob/master/docs/logo.svg?raw=True" width="200">
</p>

### Comprehensive mapping of tissue cell architecture via integrated single cell and spatial transcriptomics (cell2location model)

[![Docker image on quay.io](https://img.shields.io/badge/container-quay.io/vitkl/cell2location-brightgreen "Docker image on quay.io")](https://quay.io/vitkl/cell2location)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BayraktarLab/cell2location/blob/master/docs/notebooks/cell2location.ipynb)

Preprint: https://www.biorxiv.org/content/10.1101/2020.11.15.378125v1 

Cell2location maps the spatial distribution of cell types by integrating single-cell RNA-seq (scRNA-seq) and multi-cell spatial transcriptomic data from a given tissue (Fig 1). Cell2location leverages reference cell type signatures that are estimated from scRNA-seq profiles, for example as obtained using conventional clustering to identify cell types and subpopulations followed by estimation of average cluster gene expression profiles. Cell2location implements this estimation step based on Negative Binomial regression, which allows to robustly combine data across technologies and batches. Using these reference signatures, cell2location decomposes mRNA counts in spatial transcriptomic data, thereby estimating the relative and absolute abundance of each cell type at each spatial location (Fig 1). 

Cell2location is implemented as an interpretable hierarchical Bayesian model, (1) providing principled means to account for model uncertainty; (2) accounting for linear dependencies in cell type abundances, (3) modelling differences in measurement sensitivity across technologies, and (4) accounting for unexplained/residual variation by employing a flexible count-based error model. Finally, (5) cell2location is computationally efficient, owing to variational approximate inference and GPU acceleration. For full details and a comparison to existing approaches see our preprint (coming soon). The cell2location software comes with a suite of downstream analysis tools, including the identification of groups of cell types with similar spatial locations.


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

1. Installing dependecies and configuring environment (Method 1 (preferred) and Method 2)
2. Installing cell2location

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

#### 1. Method 1 (preferred): Create environment from file

Create `cellpymc` environment from file, which will install all the required conda and pip packages:

```bash
git clone https://github.com/BayraktarLab/cell2location.git
cd cell2location
conda env create -f environment.yml
```

Using this method will likely resolve any issues that arise when trying to create the environment manually.

#### 1. Method 2: Create conda environment manually

Create conda environment with the required packages pymc3 and scanpy:

```bash
conda create -n cellpymc python=3.7 numpy pandas jupyter leidenalg python-igraph scanpy \
louvain hyperopt loompy cmake nose tornado dill ipython bbknn seaborn matplotlib request \
mkl-service pygpu --channel bioconda --channel conda-forge
```

Do not install pymc3 and theano with conda because it will not use the system cuda (GPU drivers) and we had problems with cuda installed in the local environment, install them with pip:

```bash
conda activate cellpymc
pip install plotnine "arviz==0.10.0" "pymc3>=3.8,<3.10" torch pyro-ppl
```

### 2. Install `cell2location` package

```bash
conda activate cellpymc
pip install git+https://github.com/BayraktarLab/cell2location.git
```

Finally, to use this environment in jupyter notebook, add jupyter kernel for this environment:

```bash
conda activate cellpymc
python -m ipykernel install --user --name=cellpymc --display-name='Environment (cellpymc)'
```

## Using docker image

[![Docker image on quay.io](https://img.shields.io/badge/container-quay.io/vitkl/cell2location-brightgreen "Docker image on quay.io")](https://quay.io/vitkl/cell2location)

1. Make sure you have Docker Engine [installed](https://docs.docker.com/engine/install/). Note that you'll need root access for the installation.
   1. (recommended) If you plan to utilize GPU install [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
2. Pull docker image

       docker pull quay.io/vitkl/cell2location

3. Run docker container with GPU support

       docker run -i --rm -p 8848:8888 --gpus all quay.io/vitkl/cell2location:latest
       
   Docker is an isolated execution environment, so by default it doesn't have access to your local files. You need to mount your local directory to your docker container:
   
       docker run -v /host/directory:/container/directory -i --rm -p 8848:8888 --gpus all quay.io/vitkl/cell2location:latest

   1. For running without GPU support use
   
          docker run -i --rm -p 8848:8888 quay.io/vitkl/cell2location:latest
   
4. Go to http://127.0.0.1:8848/?token= and log in using `cell2loc` token


## Using singularity image

Singularity environments are used in the compute cluster environments (check with your local IT if Singularity is provided on your cluster). Follow the steps here to use it on your system, assuming that you need to use the GPU:

1. Download the container from our data portal:

```
wget https://cell2location.cog.sanger.ac.uk/singularity/cell2location-v0.06-alpha.sif
```

2. Submit a cluster job (LSF system) with GPU requested and start jupyter a notebook within a container (`--nv` option needed to use GPU):

```
bsub -q gpu_queue_name -M60000 \
  -R"select[mem>60000] rusage[mem=60000, ngpus_physical=1.00] span[hosts=1]"  \
  -gpu "mode=shared:j_exclusive=yes" -Is \
  /bin/singularity exec \
  --no-home  \
  --nv \
  -B /nfs/working_directory:/working_directory \
  path/to/cell2location-v0.06-alpha.sif \
  /bin/bash -c "cd /working_directory && HOME=$(mktemp -d) jupyter notebook --notebook-dir=/working_directory --NotebookApp.token='cell2loc' --ip=0.0.0.0 --port=1237 --no-browser --allow-root"
```
Replace **1)** the path to `/bin/singularity` with the one availlable on your system; **2)** the path to `/nfs/working_directory` to the directory which you need to work with (mount to the environment, `/nfs/working_directory:/working_directory`); **3)** path to the singularity image downloaded in step 1 (`path/to/cell2location-v0.06-alpha.sif`).

3. Take a note of the cluster node name `node-name` that the job started on. Go to http://node-name:1237/?token= and log in using `cell2loc` token

## Documentation and API details

User documentation is availlable on https://cell2location.readthedocs.io/en/latest/. 

The architecture of the package is briefly described [here](https://github.com/BayraktarLab/cell2location/blob/master/cell2location/models/README.md). Cell2location architecture is designed to simplify extended versions of the model that account for additional technical and biologial information. We plan to provide a tutorial showing how to add new model classes but please get in touch if you would like to contribute or build on top our package.

We also provide an experimental numpyro translation of the model. Note that the pyro translation of cell2location in this repo does not work.

## Acknowledgements 

We thank all paper authors for their contributions:
Vitalii Kleshchevnikov, Artem Shmatko, Emma Dann, Alexander Aivazidis, Hamish W King, Tong Li, Artem Lomakin, Veronika Kedlian, Mika Sarkin Jain, Jun Sung Park, Lauma Ramona, Liz Tuck, Anna Arutyunyan, Roser Vento-Tormo, Moritz Gerstung, Louisa James, Oliver Stegle, Omer Ali Bayraktar

We also thank Krzysztof Polanski, Luz Garcia Alonso, Carlos Talavera-Lopez, Ni Huang for feedback on the package, Martin Prete for dockerising cell2location and other software support.

## Common errors

#### 1. Training cell2location on GPU takes forever (>50 hours)

1. Training cell2location using `cell2location.run_cell2location()` on GPU takes forever (>50 hours). Please check that cell2location is actually using the GPU. It is crucial to add this line in your script / notebook:

```python
# this line should go before importing cell2location
os.environ["THEANO_FLAGS"] = 'device=cuda,floatX=float32,force_device=True'
import cell2location
```
which tells theano (cell2location dependency) to use the GPU before importing cell2location (or it's dependencies - theano & pymc3).
For data with 4039 locations and 10241 genes the analysis should take about 17-40 minutes depending on GPU hardware.

#### 2. `FloatingPointError: NaN occurred in optimization.`

2. `FloatingPointError: NaN occurred in optimization.` During training model parameters get into very unlikely range, resulting in division by 0 when computing gradients and breaking the optimisation:
```
FloatingPointError: NaN occurred in optimization. 
The current approximation of RV `gene_level_beta_hyp_log__`.ravel()[0] is NaN.
...
```
This usually happens when:

**A.** Numerical accuracy issues with older CUDA versions. **Solution**: use our singularity and docker images with CUDA 10.2.

**B.** The single cell reference is a very poor match to the data - reference expression signatures of cell types cannot explain most of in-situ expression. E.g. trying to map immune cell types to a tissue section that contains mostly stromal and epithelial cells. **Solution**: aim to construct a comprehensive reference.

**C.** Using cell2location in single-sample mode makes it harder to distinguish technology difference from cell abundance. **Solution**: if you have multiple expreriments try analysing them jointly in the multi-sample mode (detected automatically based on `'sample_name_col': 'sample'`).

**D.** Many genes are not expressed in the spatial data. **Solution**: try removing genes detected at low levels in spatial data.

#### 3. Theano fails to use the GPU at all (or cuDNN in particular)
3. `Can not use cuDNN on context None: cannot compile with cuDNN. ...` and other related errors. If you see these error when importing cell2location it means that you have incorrectly installed theano and it's dependencies (fix depends on the platform). Without cuDNN support training takes >3 times longer. There are **2 solutions** to this:

1. Use dockers/singularity images that are fully set up to work with the GPU (recommended).
2. Add path to system CUDA installation to the following environmental variables by adding these lines to your `.bashrc` (modify accordingly for your system):

```bash
# cuda v
cuda_v=-10.2
export CUDA_HOME=/usr/local/cuda$cuda_v
export CUDA_PATH=$CUDA_HOME
export LD_LIBRARY_PATH=/usr/local/cuda$cuda_v/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda$cuda_v/bin:$PATH
```

## FAQ

See https://github.com/BayraktarLab/cell2location/discussions
