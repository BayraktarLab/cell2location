### Make ST datasets from single-cell data
import argparse
import pickle
from sklearn.neighbors import KDTree

from cell2location.ST_simulation.ST_simulation import *

parser = argparse.ArgumentParser()
parser.add_argument('lbl_gen_file', type=str,
                    help='path to label generation pickle file')
parser.add_argument('cnt_gen_file', type=str,
                    help='path to label generation pickle file')
parser.add_argument('--tot_spots', dest='tot_spots', type=int,
                    default=1000,
                    help='Total number of spots to simulate')
parser.add_argument('--tot_regions', dest='tot_regions', type=int,
                    default=1000,
                    help='Total number of regions to simulate')
parser.add_argument('--out_dir', dest='out_dir', type=str,
                    default='/home/jovyan/rot2_BayrakLab/my_data/synthetic_ST/',
                    help='Output directory')
parser.add_argument('--assemble_id', dest='assemble_id', type=int,
                    default=1,
                    help='ID of ST assembly')

args = parser.parse_args()

lbl_gen_file = args.lbl_gen_file
count_gen_file = args.cnt_gen_file
tot_spots = args.tot_spots
tot_regions = args.tot_regions
out_dir = args.out_dir
assemble_id = args.assemble_id

### Load input data ### 

lbl_generation = pickle.load(open(lbl_gen_file, "rb"))
cnt_generation = pickle.load(open(count_gen_file, "rb"))

### EXTRACT CELL COUNT PARAMETER ###

## read spot locations and nuclei counts
df_dot = pd.read_csv('/nfs/team283/ed6/visium_chips/data/visium_beta1_segmentation/dot_df_s144600_aligned.csv',
                     header=0, index_col=0, sep=',')

df_nucl = pd.read_csv('/nfs/team283/ed6/visium_chips/data/visium_beta1_segmentation/cell_df_s144600.csv',
                      header=0, index_col=0, sep=',')

radius = 70

# count nuclei within spots
tree = KDTree(df_nucl[['x', 'y']].values)
df_dot['count'] = tree.query_radius(df_dot[['x', 'y']], radius, count_only=True)

# Select spots with at least 3 cells
cell_counts = df_dot["count"][df_dot["count"] > 2].tolist()

### EXTRACT LABEL FRACTION PARAMETER ###

sc_lbl = lbl_generation
label_counts_df = pd.DataFrame(sc_lbl["louvain"].value_counts())
label_counts_df = label_counts_df.reset_index()
label_counts_df.columns = ["louvain", "counts"]
label_counts_df["props"] = label_counts_df["counts"] / np.sum(label_counts_df["counts"])

labels = lbl_generation
cnt = cnt_generation

### GENERATE GENE-SPECIFIC SCALING FACTOR ###

gene_level_alpha = np.random.gamma(5, 5)
gene_level_beta = np.random.gamma(1, 5)
gene_level = np.random.gamma(gene_level_alpha, gene_level_beta, size=cnt.shape[1])

# scale from 0 to 1 (to coincide to fractions)
gene_level_scaled = (gene_level - min(gene_level)) / (max(gene_level) - min(gene_level))

### BUILD ST DATA ###

# Dirichlet concentration values
# making new df including only labels in the generation set
label_df_alpha = label_counts_df[label_counts_df.louvain.isin(np.unique(labels.louvain))]
alpha = np.array(label_df_alpha["props"]) + 1

# get unique labels found in single cell data
uni_labs, uni_counts = np.unique(labels,
                                 return_counts=True)

n_cells_tot = np.array(random.sample(cell_counts, k=tot_spots))  # from counts but sampling n spots

synthetic_st = assemble_st(cnt, labels, n_regions=tot_regions, n_cells_tot=n_cells_tot, alpha=alpha,
                           fraction=gene_level_scaled)

synthetic_st["regions"] = pd.DataFrame({"region": synthetic_st["regions"]}, index=synthetic_st["counts"].index)
synthetic_st["geneLevel"] = pd.DataFrame({"gene_level_scaled": gene_level_scaled}, index=cnt.columns)

### SAVE OUTPUTS ###

for k, v in synthetic_st.items():
    out_name = out_dir + "synthetic_ST_seed" + lbl_gen_file.split("_")[-1].rstrip(".p") + "_" + str(
        assemble_id) + "_" + k + ".csv"
    v.to_csv(out_name, sep=",", index=True, header=True)
