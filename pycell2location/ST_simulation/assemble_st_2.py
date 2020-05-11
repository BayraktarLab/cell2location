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

sys.path.insert(1, '/nfs/team283/ed6/cell2location/ST_simulation')
from ST_simulation import *


parser = argparse.ArgumentParser()
parser.add_argument('lbl_gen_file', type=str,
                    help='path to label generation pickle file')
parser.add_argument('cnt_gen_file', type=str,
                    help='path to label generation pickle file')
parser.add_argument('spots_comp', type=str,
                    help='path to composition csv file')
parser.add_argument('--out_dir', dest='out_dir', type=str,
                    default='/nfs/team283/ed6/simulation/lowdens_synthetic_ST/',
                    help='Output directory')
parser.add_argument('--assemble_id', dest='assemble_id', type=int,
                    default=1,
                    help='ID of ST assembly')

args = parser.parse_args()

lbl_gen_file = args.lbl_gen_file
count_gen_file = args.cnt_gen_file
spots_members_file = args.spots_comp
out_dir = args.out_dir
assemble_id = args.assemble_id
# usecols = args.usecols

### Load input data ### 

lbl_generation = pickle.load(open(lbl_gen_file, "rb"))
cnt_generation = pickle.load(open(count_gen_file, "rb"))
spots_members = pd.read_csv(spots_members_file, index_col=0)

tot_spots = spots_members.shape[1]
uni_labels = lbl_generation['annotation_1'].unique()
labels = lbl_generation
cnt = cnt_generation

### GENERATE GENE-SPECIFIC SCALING FACTOR ###

gene_level_alpha = np.random.gamma(5,5)
gene_level_beta = np.random.gamma(1,5)
gene_level = np.random.gamma(gene_level_alpha, gene_level_beta, size=cnt.shape[1])

# scale from 0 to 1 (to coincide to fractions)
gene_level_scaled = (gene_level - min(gene_level))/(max(gene_level)- min(gene_level))

### GENERATE GENE-SPECIFIC SCALING FACTOR ###

st_cnt_df = assemble_st_2(cnt, labels, spots_members, gene_level_scaled)

### SAVE OUTPUTS ###

synthetic_st = {"counts":st_cnt_df}

# usecols = args.usecols

out_name = spots_members_file.rstrip(".csv") + "_counts.csv"
st_cnt_df.to_csv(out_name, sep=",", index=True, header=True)