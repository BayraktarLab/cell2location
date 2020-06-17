### Make ST datasets from single-cell data
import sys,os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata 
import random
import collections
import scipy
import pickle
import torch as t
import torch.distributions as dists
from sklearn.neighbors import KDTree
import argparse

sys.path.insert(1, '/nfs/team238/ed6/cell2location')
from ST_simulation import *


parser = argparse.ArgumentParser()
parser.add_argument('lbl_gen_file', type=str,
                    help='path to label generation pickle file')
parser.add_argument('cnt_gen_file', type=str,
                    help='path to label generation pickle file')
parser.add_argument('design_csv', type=str,
                    help='path to design csv file')
parser.add_argument('--tot_spots', dest='tot_spots', type=int,
                    default=1000,
                    help='Total number of spots to simulate')
parser.add_argument('--out_dir', dest='out_dir', type=str,
                    default='/nfs/team283/ed6/simulation/lowdens_synthetic_ST/',
                    help='Output directory')
parser.add_argument('--assemble_id', dest='assemble_id', type=int,
                    default=1,
                    help='ID of ST assembly')

args = parser.parse_args()

lbl_gen_file = args.lbl_gen_file
count_gen_file = args.cnt_gen_file
design_file = args.design_csv
tot_spots = args.tot_spots
out_dir = args.out_dir
assemble_id = args.assemble_id

### Load input data ### 

lbl_generation = pickle.load(open(lbl_gen_file, "rb"))
cnt_generation = pickle.load(open(count_gen_file, "rb"))

uni_labels = lbl_generation['annotation_1'].unique()
labels = lbl_generation
cnt = cnt_generation

### GENERATE GENE-SPECIFIC SCALING FACTOR ###

gene_level_alpha = np.random.gamma(5,5)
gene_level_beta = np.random.gamma(1,5)
gene_level = np.random.gamma(gene_level_alpha, gene_level_beta, size=cnt.shape[1])

# scale from 0 to 1 (to coincide to fractions)
gene_level_scaled = (gene_level - min(gene_level))/(max(gene_level)- min(gene_level))

design_df = pd.read_csv(design_file, index_col=0)

### Assemble cell type composition
spots_members = assemble_ct_composition(design_df, tot_spots, ncells_scale=1)

# st_cnt_df = assemble_st_2(cnt, labels, spots_members, gene_level_scaled)

### SAVE OUTPUTS ###

synthetic_st = {"composition":spots_members, "design":design_df}

for k,v in synthetic_st.items():
    out_name = out_dir + "synthetic_ST_seed" + lbl_gen_file.split("_")[-1].rstrip(".p") + "_" + str(assemble_id) + "_" + k + ".csv"
    v.to_csv(out_name, sep=",", index=True, header=True)
