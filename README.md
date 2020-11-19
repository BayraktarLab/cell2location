<p align="center">
   <img src="https://github.com/BayraktarLab/cell2location/blob/master/docs/logo.svg" width="200">
</p>

### Comprehensive mapping of tissue cell architecture via integrated single cell and spatial transcriptomics (cell2location model)

[![Docker Repository on Quay](https://quay.io/repository/vitkl/cell2location/status "Docker Repository on Quay")](https://quay.io/repository/vitkl/cell2location)

Cell2location maps the spatial distribution of cell types by integrating single-cell RNA-seq (scRNA-seq) and multi-cell spatial transcriptomic data from a given tissue (Fig 1). Cell2location leverages reference cell type signatures that are estimated from scRNA-seq profiles, for example as obtained using conventional clustering to identify cell types and subpopulations followed by estimation of average cluster gene expression profiles. Cell2location implements this estimation step based on Negative Binomial regression, which allows to robustly combine data across technologies and batches. Using these reference signatures, cell2location decomposes mRNA counts in spatial transcriptomic data, thereby estimating the relative and absolute abundance of each cell type at each spatial location (Fig 1). 

Cell2location is implemented as an interpretable hierarchical Bayesian model, (1) providing principled means to account for model uncertainty; (2) accounting for linear dependencies in cell type abundances, (3) modelling differences in measurement sensitivity across technologies, and (4) accounting for unexplained/residual variation by employing a flexible count-based error model. Finally, (5) cell2location is computationally efficient, owing to variational approximate inference and GPU acceleration. For full details and a comparison to existing approaches see our preprint (coming soon). The cell2location software comes with a suite of downstream analysis tools, including the identification of groups of cell types with similar spatial locations.


![Fig1](docs/images/Fig1_v2.png)   
Overview of the spatial mapping approach and the workflow enabled by cell2location. From left to right: Single-cell RNA-seq and spatial transcriptomics profiles are generated from the same tissue (1). Cell2location takes scRNA-seq derived cell type reference signatures and spatial transcriptomics data as input (2, 3). The model then decomposes spatially resolved multi-cell RNA counts matrices into the reference signatures, thereby establishing a spatial mapping of cell types (4).    

## Usage and Tutorials

Tutorials covering the estimation of expresson signatures of reference cell types (1/3), spatial mapping with cell2location (2/3) and the downstream analysis (3/3) can be found here: https://cell2location.readthedocs.io/en/latest/

There are 2 ways to install and use our package: setup your [own conda environment](https://github.com/BayraktarLab/cell2location#installation-of-dependecies-and-configuring-environment) or use the [singularity](https://github.com/BayraktarLab/cell2location#using-singularity-image) and [docker](https://github.com/BayraktarLab/cell2location#using-docker-image) images (recommended). See below for details.

You can also try cell2location on [Google Colab](https://colab.research.google.com/github/BayraktarLab/cell2location/blob/master/docs/notebooks/cell2location_short_demo_colab.ipynb) on a smaller data subset containing somatosensory cortex.

Please report bugs and feedback via https://github.com/BayraktarLab/cell2location/issues .

## Configure your own conda environment

1. Installation of dependecies and configuring environment (Method 1 and Method 2)
2. Installation of cell2location

Prior to installing cell2location package you need to install miniconda and create a conda environment containing pymc3 and theano ready for use on GPU. Follow the steps below:

If you do not have conda please install Miniconda first:

```bash
cd /path/to/software
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use prefix /path/to/software/miniconda3
```

#### 1. Method 1: Create conda environment manually

Create conda environment with the required packages pymc3 and scanpy:

```bash
conda create -n cellpymc python=3.7 numpy pandas jupyter leidenalg python-igraph scanpy \
louvain hyperopt loompy cmake nose tornado dill ipython bbknn seaborn matplotlib request \
mkl-service pygpu --channel bioconda --channel conda-forge
```

Do not install pymc3 and theano with conda because it will not use the system cuda (GPU drivers) and we had problems with cuda installed in the local environment, install them with pip:

```bash
pip install plotnine pymc3 torch pyro-ppl
```

#### 1. Method 2: Create environment from file

Create `cellpymc` environment from file, which will install all the required conda and pip packages:

```bash
git clone https://github.com/BayraktarLab/cell2location.git
cd cell2location
conda env create -f environment.yml
```

### 2. Install `cell2location` package

```bash
pip install git+https://github.com/BayraktarLab/cell2location.git
```

## Using docker image

[![Docker Repository on Quay](https://quay.io/repository/vitkl/cell2location/status "Docker Repository on Quay")](https://quay.io/repository/vitkl/cell2location)

1. Make sure you have Docker Engine [installed](https://docs.docker.com/engine/install/). Note that you'll need root access for the installation.
   1. (recommended) If you plan to utilize GPU install [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
2. Pull docker image

       docker pull quay.io/vitkl/cell2location

3. Run docker container

       docker run -i --rm -p 8848:8888 quay.io/vitkl/cell2location:latest

   1. (recommended) For running with GPU support use
   
          docker run -i --rm -p 8848:8888 --gpus all quay.io/vitkl/cell2location:latest
   
4. Go to http://127.0.0.1:8848/?token= and log in using `cell2loc` token


## Using singularity image

Singularity environments are used in the compute cluster environments (check with your local IT if Singularity is provided on your cluster). Follow the steps here to use it on your system, assuming that you need to use the GPU:

1. Download the container from our data portal:

```
wget https://cell2location.cog.sanger.ac.uk/singularity/cell2location-20201116.sif
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
  path/to/cell2location-20201116.sif \
  /bin/bash -c "cd /working_directory && HOME=$(mktemp -d) jupyter notebook --notebook-dir=/working_directory --NotebookApp.token='cell2loc' --ip=0.0.0.0 --port=1237 --no-browser --allow-root"
```
Replace **1)** the path to `/bin/singularity` with the one availlable on your system; **2)** the path to `/nfs/working_directory` to the directory which you need to work with (mount to the environment, `/nfs/working_directory:/working_directory`); **3)** path to the singularity image downloaded in step 1 (`path/to/cell2location-20201116.sif`).

3. Take a note of the cluster node name `node-name` that the job started on. Go to http://node-name:1237/?token= and log in using `cell2loc` token

## Documentation and API details

User documentation is availlable on https://cell2location.readthedocs.io/en/latest/. 

The architecture of the package is briefly described [here](https://github.com/BayraktarLab/cell2location/blob/master/cell2location/models/README.md). Cell2location architecture is designed to simplify extended versions of the model that account for additional technical and biologial information. We plan to provide a tutorial showing how to add new model classes but please get in touch if you would like to contribute or build on top our package.

## Documentation and API details

We thank all paper authors for their contributions:
Vitalii Kleshchevnikov, Artem Shmatko, Emma Dann, Alexander Aivazidis, Hamish W King, Tong Li, Artem Lomakin, Veronika Kedlian, Mika Sarkin Jain, Jun Sung Park, Lauma Ramona, Liz Tuck, Anna Arutyunyan, Roser Vento-Tormo, Moritz Gerstung, Louisa James, Oliver Stegle, Omer Ali Bayraktar

We also thank Krzysztof Polanski, Luz Garcia Alonso, Carlos Talavera-Lopez for feedback on the package, Martin Prete for dockerising cell2location and other software support.
