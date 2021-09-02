## Using docker image

[![Docker image on Quay](https://quay.io/repository/vitkl/cell2location/status "Docker Repository on Quay")](https://quay.io/repository/vitkl/cell2location)

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
