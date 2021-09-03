## Installation of pymc3 version with GPU support

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