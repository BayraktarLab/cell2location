### SPLIT SINGLE-CELL DATASET IN GENERATION AND VALIDATION SET ###

import sys,os
import pickle
import pandas as pd
import numpy as np
import scanpy as sc
import anndata 
import random
import collections
import scipy
import torch as t
import torch.distributions as dists
from sklearn.neighbors import KDTree

adata_raw = anndata.read_mtx('/nfs/team205/vk7/sanger_projects/large_data/mouse_viseum_snrna/rawdata/all.mtx').T

adata_snrna = anndata.read_h5ad("/nfs/team283/ed6/processed_data/visium_st_beta/snRNA_s144600_preprocessed_20200109.h5ad")

## Cell type annotations
labels = pd.read_csv('/nfs/team283/ed6/processed_data/visium_st_beta/snRNA_annotation_20200229.csv', index_col=0)

# ## Add cell type labels as columns in adata.obs
# adata_snrna = adata_snrna_raw[labels.index,]
# adata_snrna.obs = pd.concat([labels, adata_snrna_raw.obs], axis=1)

# add cell names
obs_id = pd.read_csv('/nfs/team205/vk7/sanger_projects/large_data/mouse_viseum_snrna/rawdata/all_cells.txt')
obs = obs_id['cell_id'].str.split(pat="_", expand = True)
obs.columns = ["sample_id", "barcode"]
obs['cell_id'] = obs_id['cell_id']
obs.index = obs['cell_id']
adata_raw.obs_names = obs['cell_id'].tolist()
obs['cell_id'] = None

obs_data = pd.read_csv('/nfs/team205/vk7/sanger_projects/large_data/mouse_viseum_snrna/zeisel_integrated_meta.tsv', sep = "\t")
obs_data.index = obs_data['cell_id']
obs_data = obs_data.loc[adata_raw.obs_names,]

adata_raw.obs = obs_data

# add ensembl row names
var = pd.read_csv('/nfs/team205/vk7/sanger_projects/large_data/mouse_viseum_snrna/rawdata/all_genes.txt')
var.columns = ['ENSEMBL']
var.index = var['ENSEMBL']
adata_raw.var_names = var['ENSEMBL']

var = pd.read_csv('/nfs/team205/vk7/sanger_projects/large_data/mouse_viseum_snrna/rawdata/filtered_feature_bc_matrix/5705STDY8058280/features.tsv.gz',
                 sep='\t', header=None)
var.columns = ['ENSEMBL', 'SYMBOL', 'weird']
var.index = var['ENSEMBL']
var = var.loc[var.index,:]
adata_raw.var = var
adata_raw.var_names = var['ENSEMBL']

# Select genes used for clustering
adata_raw = adata_raw[:,[x for x in adata_raw.var_names if x in adata_snrna.var_names]]

adata_df = pd.DataFrame(adata_raw.X.T.toarray(), columns=adata_raw.obs_names, index=adata_raw.var_names)
adata_df = adata_df.T
adata_df.index.name="cell"


### Subset to cells with label ###
adata_df = adata_df.loc[labels.index,:]

### Split generation and validation set ###

sc_cnt = adata_df
sc_lbl = pd.DataFrame(labels["annotation_1"])

# match count and label data
inter = sc_cnt.index.intersection(sc_lbl.index)

sc_lbl = sc_lbl.loc[inter,:]
sc_cnt = sc_cnt.loc[inter,:]

labels = sc_lbl.iloc[:,0].values

# get unique labels
uni_labs, uni_counts = np.unique(labels,return_counts = True)

# only keep types with more than 200 cells
keep_types = uni_counts > 200
keep_cells = np.isin(labels, uni_labs[keep_types])

labels = labels[keep_cells]
sc_cnt = sc_cnt.iloc[keep_cells,:]
sc_lbl = sc_lbl.iloc[keep_cells,:]

uni_labs, uni_counts = np.unique(labels,return_counts = True)
n_types = uni_labs.shape[0]

seeds = random.sample(range(1000), 3)

for seed in seeds:
    random.seed(seed)
    print("Seed " + str(seed))
    # get member indices for each set
    idx_generation = []
    idx_validation = []
    for z in range(n_types):
        tmp_idx = np.where(labels == uni_labs[z])[0]
        n_generation = int(round(tmp_idx.shape[0] / 2 ))
        smp_gen = random.sample(list(tmp_idx), k=n_generation)
        smp_val = tmp_idx[np.isin(tmp_idx, smp_gen, invert=True)]
        idx_generation += smp_gen
        idx_validation += smp_val.tolist()
    idx_generation.sort()
    idx_validation.sort()
    # make sure no members overlap between sets
    assert len(set(idx_generation).intersection(set(idx_validation))) == 0, \
            "validation and genreation set are not orthogonal"
    # assemble sets from indices
    cnt_validation = sc_cnt.iloc[idx_validation,:]
    cnt_generation = sc_cnt.iloc[idx_generation,:]
    lbl_validation = sc_lbl.iloc[idx_validation,:]
    lbl_generation = sc_lbl.iloc[idx_generation,:]
    pickle.dump(lbl_generation, open("/nfs/team283/ed6/simulation/lowdens_synthetic_ST/labels_generation_" + str(seed) + ".p", "wb"))
    pickle.dump(cnt_generation, open("/nfs/team283/ed6/simulation/lowdens_synthetic_ST/counts_generation_" + str(seed) + ".p", "wb"))
    pickle.dump(lbl_validation, open("/nfs/team283/ed6/simulation/lowdens_synthetic_ST/labels_validation_" + str(seed) + ".p", "wb"))
    pickle.dump(cnt_validation, open("/nfs/team283/ed6/simulation/lowdens_synthetic_ST/counts_validation_" + str(seed) + ".p", "wb"))

