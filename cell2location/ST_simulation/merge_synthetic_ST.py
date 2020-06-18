### Merge synthetic ST data ###
import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('outdir', type=str,
                    help='path to synthetic ST directory')
parser.add_argument('seed', type=int,
                    help='random seed for generation')
args = parser.parse_args()

outdir = args.outdir
seed = args.seed


def read_synthetic_ST(outdir, seed, id):
    cnt_file = 'synthetic_ST_seed{0}_{1}_composition_counts.csv'.format(seed, id)
    comp_file = cnt_file.split("_counts")[0] + ".csv"
    cnt_df = pd.read_csv(os.path.join(outdir, cnt_file), index_col=0)
    comp_df = pd.read_csv(os.path.join(outdir, comp_file), index_col=0)
    return (cnt_df, comp_df)


design_file = [x for x in os.listdir(outdir) if 'design.csv' in x and str(seed) in x][0]
seed_files = [x for x in os.listdir(outdir) if str(seed) in x and x.endswith("counts.csv")]
ids = [x.split("_")[3] for x in seed_files]

cnt_df, comp_df = read_synthetic_ST(outdir, seed, ids[0])
for id in ids[1:]:
    cnt_df1, comp_df1 = read_synthetic_ST(outdir, seed, id)
    if cnt_df1.shape[1] == cnt_df.shape[1]:
        cnt_df = pd.concat([cnt_df, cnt_df1])
    if comp_df.shape[0] == comp_df.shape[0]:
        comp_df = pd.concat([comp_df, comp_df1], 1)

cnt_df.reset_index(drop=True, inplace=True)
cnt_df.index = ["Spotx" + str(x) for x in cnt_df.index]
comp_df.columns = ["Spotx" + str(x) for x in range(comp_df.shape[1])]

cnt_df.to_csv(os.path.join(outdir, design_file.split("design")[0] + "counts.csv"))
comp_df.to_csv(os.path.join(outdir, design_file.split("design")[0] + "composition.csv"))
