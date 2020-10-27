BootStrap: docker
From: quay.io/cellgeni/cell2location:0.1

%help
    cell2location: High-throughput spatial mapping of cell types.

%labels
    Maintainer Vitalii Kleshchevnikov <vitalii.kleshchevnikov@sanger.ac.uk>
    Version v0.1

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export PORT=8888    
    export PASSWORD=cell2loc

%runscript
     exec /bin/bash -c "jupyter notebook --notebook-dir=/notebooks --NotebookApp.token='$PASSWORD' --ip=0.0.0.0 --port=$PORT --no-browser --allow-root"

