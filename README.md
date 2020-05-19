# cell2location
### High throughput mapping of single cell reference cell types and expression programmes to spatial sequencing data (aggregate across cells in a small tissue region)

## Installation

Prior to intalling cell2location package you need to install miniconda and create a conda environment containing pymc3 and theano ready for use on GPU. Follow the steps below:   

If you do not have conda please install Miniconda first:   
```
cd /path/to/software
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use prefix /path/to/software/miniconda3
```

Install packages needed for pymc3 and scanpy to work:   
```
### essential
# Create conda envirnment that can work with jupyter & GPUs
conda create -n cellpymc python=3.7 numpy pandas jupyter leidenalg python-igraph scanpy louvain hyperopt loompy cmake nose tornado dill ipython bbknn seaborn matplotlib request mkl-service pygpu --channel bioconda --channel conda-forge

# Do not install pymc3 and theano with conda because it will not use the system cuda and we had problems with cuda installed in the local environment
pip install plotnine pymc3

### extras - you do not need this for standars use
# If you use pyro implementatation you also need to install pyro-ppl and torch
pip install torch pyro-ppl

# If you use tensorflow implementation (not implemented yet) also install
pip install tensorboard keras tensorflow-gpu
```

Now install `cell2location` package:   
```
pip install git+https://github.com/BayraktarLab/cell2location.git
```

## Usage

See [cell2location short demo Jupyter notebook](https://github.com/BayraktarLab/cell2location/blob/master/notebooks/cell2location_short_demo.ipynb) for usage example.   

See `pyro` branch for pyro implementation of the model.  

## API details

Models are implemented as dedicated python classes organised in an [inheritance hierarchy](https://github.com/BayraktarLab/cell2location/blob/master/pycell2location/models/README.md) to enable reusing methods between pymc3 and pyro models as well as models that answer distinct questions.  
