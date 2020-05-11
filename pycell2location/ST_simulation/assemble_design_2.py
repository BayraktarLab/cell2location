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
tot_spots = args.tot_spots
out_dir = args.out_dir
assemble_id = args.assemble_id

### Load input data ### 

lbl_generation = pickle.load(open(lbl_gen_file, "rb"))
cnt_generation = pickle.load(open(count_gen_file, "rb"))

uni_labels = lbl_generation['annotation_1'].unique()
labels = lbl_generation
cnt = cnt_generation

### Define uniform VS regional cell types
uniform_ct = np.random.choice([0, 1], size=len(uni_labels), p=[0.8,0.2])

#### Define low VS high density cell types
uni_low = np.random.choice([0, 1], size=len(uni_labels[uniform_ct==1]))
reg_low = np.random.choice([0, 1], size=len(uni_labels[uniform_ct==0]))

design_df = pd.DataFrame({'uniform':uniform_ct}, index=uni_labels)

design_df['density'] = np.nan
design_df.loc[design_df.index[design_df.uniform==1],'density'] = uni_low
design_df.loc[design_df.index[design_df.uniform==0],'density'] = reg_low

### Generate no of spots per cell type
mean_unif = round((tot_spots / 100) * 80)
mean_reg = round((tot_spots / 100) * 10)

unif_nspots = np.round(np.random.normal(mean_unif, tot_spots/100, size=sum(design_df.uniform==1)))
reg_nspots = np.round(np.random.normal(mean_reg, tot_spots/100, size=sum(design_df.uniform==0)))

design_df['nspots'] = np.nan
design_df.loc[design_df.index[design_df.uniform==1],'nspots'] = unif_nspots
design_df.loc[design_df.index[design_df.uniform==0],'nspots'] = reg_nspots

### Generate avg density per spot per cell type

mean_high = 15
mean_low = 5

low_ncells_mean = np.round(np.random.normal(mean_low, 1, size=sum(design_df.density==1)))
high_ncells_mean = np.round(np.random.normal(mean_high, 1, size=sum(design_df.density==0)))

design_df['mean_ncells'] = np.nan
design_df.loc[design_df.index[design_df.density==1],'mean_ncells'] = low_ncells_mean
design_df.loc[design_df.index[design_df.density==0],'mean_ncells'] = high_ncells_mean

out_name = out_dir + "synthetic_ST_seed" + lbl_gen_file.split("_")[-1].rstrip(".p") + "_" + "design" + ".csv"
design_df.to_csv(out_name, sep=",", index=True, header=True)