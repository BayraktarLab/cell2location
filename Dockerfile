# base image maintained by the NVIDIA CUDA Installer Team - https://hub.docker.com/r/nvidia/cuda/
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

LABEL version="0.03"
LABEL maintainer="Vitalii Kleshchevnikov <vitalii.kleshchevnikov@sanger.ac.uk>"
LABEL description="High-throughput spatial mapping of cell types."

# install os packages
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        curl \
        unzip \
        g++ \
        wget \
        vim \
        ca-certificates \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# see http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# cuda v
RUN echo "cuda_v=-10.2" >> ~/.bashrc
# add CUDA specifications for GPU nodes
RUN echo "export PATH=/usr/local/cuda$cuda_v/bin:$PATH" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda$cuda_v/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
RUN echo "export CUDA_HOME=/usr/local/cuda$cuda_v" >> ~/.bashrc
RUN echo "export CUDA_PATH=$CUDA_HOME" >> ~/.bashrc

# install miniconda3 - https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
RUN /opt/conda/condabin/conda init bash

# create conda environment yaml file
COPY environment.yml /tmp/
RUN /opt/conda/condabin/conda env create -f /tmp/environment.yml \
    && echo "conda activate cellpymc" >> ~/.bashrc \
    && /opt/conda/condabin/conda clean --all --yes --quiet
ENV PATH /opt/conda/envs/cellpymc/bin:/opt/conda/bin:$PATH

# install cell2location and add cellpymc kernel for jupyter environment 
RUN /bin/bash -c "pip install git+https://github.com/BayraktarLab/cell2location.git" 
RUN python -m ipykernel install --user --name cellpymc --display-name "Container (cellpymc)"

# copy notebooks to the image
COPY docs/notebooks notebooks
RUN /bin/bash -c "jupyter trust /notebooks/*.ipynb";

# launch jupyter
CMD ["jupyter", "notebook", \
    "--notebook-dir=/notebooks", \
    "--NotebookApp.token='cell2loc'", \
    "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
EXPOSE 8888
