wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
conda install -q -y --prefix /usr/local python=3.10 numpy pandas jupyter leidenalg python-igraph scanpy louvain hyperopt loompy cmake nose tornado dill ipython bbknn seaborn matplotlib request --channel bioconda --channel conda-forge
git clone https://github.com/BayraktarLab/cell2location.git cell2location_repo
cd cell2location_repo && pip install -q --prefix /usr/local .
