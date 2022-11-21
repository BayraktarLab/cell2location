<p align="center">
   <img src="https://github.com/BayraktarLab/cell2location/blob/master/docs/logo.svg?raw=True" width="200">
</p>

### Comprehensive mapping of tissue cell architecture via integrated single cell and spatial transcriptomics (cell2location model)

[![Stars](https://img.shields.io/github/stars/BayraktarLab/cell2location?logo=GitHub&color=yellow)](https://github.com/BayraktarLab/cell2location/stargazers)
![Build Status](https://github.com/BayraktarLab/cell2location/actions/workflows/test.yml/badge.svg?event=push)
[![Documentation Status](https://readthedocs.org/projects/cell2location/badge/?version=latest)](https://cell2location.readthedocs.io/en/stable/?badge=latest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BayraktarLab/cell2location/blob/master/docs/notebooks/cell2location_tutorial.ipynb)
[![Docker image on quay.io](https://img.shields.io/badge/container-quay.io/vitkl/cell2location-brightgreen "Docker image on quay.io")](https://quay.io/vitkl/cell2location) 

If you use cell2location please cite our paper: 

Kleshchevnikov, V., Shmatko, A., Dann, E. et al. Cell2location maps fine-grained cell types in spatial transcriptomics. Nat Biotechnol (2022). https://doi.org/10.1038/s41587-021-01139-4
https://www.nature.com/articles/s41587-021-01139-4

Please note that cell2locations requires 2 user-provided hyperparameters (N_cells_per_location and detection_alpha) - for detailed guidance on setting these hyperparameters and their impact see [the flow diagram and the note](https://github.com/BayraktarLab/cell2location/blob/master/docs/images/Note_on_selecting_hyperparameters.pdf). Many real datasets (especially human) show within-slide variability in RNA detection sensitivity - requiring you to try both recommended settings of the `detection_alpha` parameter: `detection_alpha=200` for low within-slide technical variability and `detection_alpha=20` for high within-slide technical variability.

Cell2location is a principled Bayesian model that can resolve fine-grained cell types in spatial transcriptomic data and create comprehensive cellular maps of diverse tissues. Cell2location accounts for technical sources of variation and borrows statistical strength across locations, thereby enabling the integration of single cell and spatial transcriptomics with higher sensitivity and resolution than existing tools. This is achieved by estimating which combination of cell types in which cell abundance could have given the mRNA counts in the spatial data, while modelling technical effects (platform/technology effect, contaminating RNA, unexplained variance).

<p align="center">
   <img src="https://github.com/BayraktarLab/cell2location/blob/master/docs/images/Fig1_v2_white_bg.png?raw=True">
</p>
Overview of the spatial mapping approach and the workflow enabled by cell2location. From left to right: Single-cell RNA-seq and spatial transcriptomics profiles are generated from the same tissue (1). Cell2location takes scRNA-seq derived cell type reference signatures and spatial transcriptomics data as input (2, 3). The model then decomposes spatially resolved multi-cell RNA counts matrices into the reference signatures, thereby establishing a spatial mapping of cell types (4).    

## Usage and Tutorials

The tutorial covering the estimation of expresson signatures of reference cell types, spatial mapping with cell2location and the downstream analysis can be found here and tried on [Google Colab](https://colab.research.google.com/github/BayraktarLab/cell2location/blob/master/docs/notebooks/cell2location_tutorial.ipynb): https://cell2location.readthedocs.io/en/latest/

Please report bugs via https://github.com/BayraktarLab/cell2location/issues and ask any usage questions about [cell2location](https://discourse.scverse.org/c/ecosytem/cell2location/42), [scvi-tools](https://discourse.scverse.org/c/help/scvi-tools/7) or [Visium data](https://discourse.scverse.org/c/general/visium/32) in scverse community discourse.

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

Before installing cell2location and it's dependencies, it could be necessary to make sure that you are creating a fully isolated conda environment by telling python to NOT use user site for installing packages by running this line before creating conda environment and every time before activatin conda environment in a new terminal session:

```bash
export PYTHONNOUSERSITE="literallyanyletters"
```


## Documentation and API details

User documentation is availlable on https://cell2location.readthedocs.io/en/latest/. 

Cell2location architecture is designed to simplify extended versions of the model that account for additional technical and biologial information. We plan to provide a tutorial showing how to add new model classes but please get in touch if you would like to contribute or build on top our package.

## Acknowledgements 

We thank all paper authors for their contributions:
Vitalii Kleshchevnikov, Artem Shmatko, Emma Dann, Alexander Aivazidis, Hamish W King, Tong Li, Artem Lomakin, Veronika Kedlian, Mika Sarkin Jain, Jun Sung Park, Lauma Ramona, Liz Tuck, Anna Arutyunyan, Roser Vento-Tormo, Moritz Gerstung, Louisa James, Oliver Stegle, Omer Ali Bayraktar

We also thank Pyro developers (Fritz Obermeyer, Martin Jankowiak), Krzysztof Polanski, Luz Garcia Alonso, Carlos Talavera-Lopez, Ni Huang for feedback on the package, Martin Prete for dockerising cell2location and other software support.

## FAQ

See https://github.com/BayraktarLab/cell2location/discussions

## Future development and experimental features
Future developments of cell2location are focused on 1) scalability to 100k-mln+ locations using amortised inference of cell abundance (same ideas as used in VAE), 2) extending cell2location to related spatial analysis tasks that require modification of the model (such as using cell type hierarchy information), and 3) incorporating features presented by more recently proposed methods (such as CAR spatial proximity modelling). We are also experimenting with Numpyro and JAX (https://github.com/vitkl/cell2location_numpyro).

## Tips

### Conda environment for A100 GPUs

```bash
export PYTHONNOUSERSITE="literallyanyletters"
conda create -y -n test_scvi16_cuda113 python=3.9
conda activate test_scvi16_cuda113
conda install -y -c anaconda hdf5 pytables git
pip install scvi-tools
pip install git+https://github.com/BayraktarLab/cell2location.git#egg=cell2location[tutorials]
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html
conda activate test_scvi16_cuda113
python -m ipykernel install --user --name=test_scvi16_cuda113 --display-name='Environment (test_scvi16_cuda113)'
```

### Issues with package version mismatches often originate from python user site rather than conda environment being used to install a subset of packages

Before installing cell2location and it's dependencies, it could be necessary to make sure that you are creating a fully isolated conda environment by telling python to NOT use user site for installing packages by running this line before creating conda environment and every time before activatin conda environment in a new terminal session:

```bash
export PYTHONNOUSERSITE="literallyanyletters"
```

### Useful code for reading and combining multiple Visium sections

Keeping info on distinct sections in a csv file (Google Sheet).

```python
sample_annot = pd.read_csv('./sample_annot.csv')

from glob import glob
sample_annot['path'] = pd.Series(
    glob(f'{sp_data_folder}*'),
    index=[sub('^.+WTSI_', '', sub('_GRCh38-2020-A$', '', i)) for i in glob(f'{sp_data_folder}*')]
)[sample_annot['Sample_ID']].values
import os
sample_annot['file'] = [os.path.basename(i) for i in sample_annot['path']]

sample_annot['Sample_ID'].unique()
```

Reading and concatenating samples.

```python
def read_and_qc(sample_name, file, path=sp_data_folder):
    """
    Read one Visium file and add minimum metadata and QC metrics to adata.obs
    NOTE: var_names is ENSEMBL ID as it should be, you can always plot with sc.pl.scatter(gene_symbols='SYMBOL')
    """
    
    adata = sc.read_visium(path + str(file) +'/',
                           count_file='filtered_feature_bc_matrix.h5',
                           load_images=True)
    adata.obs['sample'] = sample_name
    adata.var['SYMBOL'] = adata.var_names
    adata.var.rename(columns={'gene_ids': 'ENSEMBL'}, inplace=True)
    adata.var_names = adata.var['ENSEMBL']
    adata.var.drop(columns='ENSEMBL', inplace=True)
    
    # just in case there are non-unique ENSEMBL IDs
    adata.var_names_make_unique()

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.var['mt'] = [gene.startswith('mt-') for gene in adata.var['SYMBOL']]
    adata.obs['mt_frac'] = adata[:, adata.var['mt'].tolist()].X.sum(1).A.squeeze()/adata.obs['total_counts']
    
    # add sample name to obs names
    adata.obs["sample"] = [str(i) for i in adata.obs['sample']]
    adata.obs_names = 's' + adata.obs["sample"] \
                          + '_' + adata.obs_names
    adata.obs.index.name = 'spot_id'
    
    file = list(adata.uns['spatial'].keys())[0]
    adata.uns['spatial'][sample_name] = adata.uns['spatial'][file].copy()
    del adata.uns['spatial'][file]
    print(adata.uns['spatial'].keys())
    
    return adata

def read_all_and_qc(
    sample_annot, Sample_ID_col, file_col, sp_data_folder, 
    count_file='filtered_feature_bc_matrix.h5',
):
    """
    Read and concatenate all Visium files.
    """
    # read first sample
    adata = read_and_qc(
        sample_annot[Sample_ID_col][0], sample_annot[file_col][0], 
        path=sp_data_folder
    ) 

    # read the remaining samples
    slides = {}
    for i, s in enumerate(sample_annot[Sample_ID_col][1:]):
        adata_1 = read_and_qc(s, sample_annot[file_col][i], path=sp_data_folder) 
        slides[str(s)] = adata_1

    adata_0 = adata.copy()

    # combine individual samples
    #adata = adata.concatenate(list(slides.values()), index_unique=None)
    adata = adata.concatenate(
        list(slides.values()),
        batch_key="sample",
        uns_merge="unique",
        batch_categories=sample_annot[Sample_ID_col], 
        index_unique=None
    )

    sample_annot.index = sample_annot[Sample_ID_col]
    for c in sample_annot.columns:
        sample_annot.loc[:, c] = sample_annot[c].astype(str)
    adata.obs[sample_annot.columns] = sample_annot.reindex(index=adata.obs['sample']).values
    
    return adata
    
adata = read_all_and_qc(
    sample_annot=sample_annot, 
    Sample_ID_col='Sample_ID', 
    file_col='file', 
    sp_data_folder=sp_data_folder, 
    count_file='filtered_feature_bc_matrix.h5',
)

adata_incl_nontissue = read_all_and_qc(
    sample_annot=sample_annot, 
    Sample_ID_col='Sample_ID', 
    file_col='file', 
    sp_data_folder=sp_data_folder, 
    count_file='raw_feature_bc_matrix.h5',
)
```
