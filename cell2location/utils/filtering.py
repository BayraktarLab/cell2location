import matplotlib.pyplot as plt
import numpy as np


def check_gene_filter(adata, nonz_mean_cutoff=np.log10(1.12), cell_count_cutoff=np.log10(15), cell_count_cutoff2=None):
    r"""Plot the gene filter given a set of gene cutoffs and give
     the resulting shape. To not be filtered out, a gene has
     to either be expressed in at least in cell_count_cutoff2 cells,
     or it has to be expressed in at least cell_count_cutoff cells AND
     have a mean of non-zero values above nonz_mean_cutoff.

    Parameters
    ----------
    adata :
        anndata object with single cell / nucleus data
    nonz_mean_cutoff :
        log10-transformed cutoff for mean of non-zero values
    cell_count_cutoff :
        log10-transformed cell cutoff for lower cutoff (used in
        combination with nonz_mean_cutoff)
    cell_count_cutoff2 :
        log10-transformed cell cutoff for highest cutoff (genes
        expressed in at least this amount of cells will be included).
        If None, this will default to adata.shape[0] * 0.05.

    Returns
    -------
    None
    """

    adata.var["n_cells"] = np.array((adata.X > 0).sum(0)).flatten()
    adata.var["nonz_mean"] = np.array(adata.X.sum(0)).flatten() / adata.var["n_cells"]

    if cell_count_cutoff2 is None:
        cell_count_cutoff2 = np.log10(adata.shape[0] * 0.05)

    fig, ax = plt.subplots()
    ax.hist2d(
        np.log10(adata.var["nonz_mean"]),
        np.log10(adata.var["n_cells"]),
        bins=100,
        norm=matplotlib.colors.LogNorm(),
        range=[[0, 0.5], [1, 4.5]],
    )
    ax.axvspan(0, nonz_mean_cutoff, ymin=0.0, ymax=(cell_count_cutoff2 - 1) / 3.5, color="darkorange", alpha=0.3)
    ax.axvspan(
        nonz_mean_cutoff,
        np.max(np.log10(adata.var["nonz_mean"])),
        ymin=0.0,
        ymax=(cell_count_cutoff - 1) / 3.5,
        color="darkorange",
        alpha=0.3,
    )
    plt.vlines(nonz_mean_cutoff, cell_count_cutoff, cell_count_cutoff2, color="darkorange")
    plt.hlines(cell_count_cutoff, nonz_mean_cutoff, 1, color="darkorange")
    plt.hlines(cell_count_cutoff2, 0, nonz_mean_cutoff, color="darkorange")
    plt.xlabel("Mean non-zero expression level of gene (log)")
    plt.ylabel("Number of cells expressing gene (log)")
    plt.title(
        "Gene filter - resulting shape: "
        + str(
            adata[
                :,
                (np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff2))
                | (
                    np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff)
                    & np.array(np.log10(adata.var["nonz_mean"]) > nonz_mean_cutoff)
                ),
            ].shape
        )
    )
    plt.show()


def apply_gene_filter(adata, nonz_mean_cutoff=np.log10(1.12), cell_count_cutoff=np.log10(15), cell_count_cutoff2=None):
    r"""Applies a gene filter given a set of gene cutoffs. To not be
     filtered out, a gene has to either be expressed in at least in
     cell_count_cutoff2 cells, or it has to be expressed in at least
     cell_count_cutoff cells AND have a mean of non-zero values above
     nonz_mean_cutoff.

    Parameters
    ----------
    adata :
        anndata object with single cell / nucleus data
    nonz_mean_cutoff :
        log10-transformed cutoff for mean of non-zero values
    cell_count_cutoff :
        log10-transformed cell cutoff for lower cutoff (used in
        combination with nonz_mean_cutoff)
    cell_count_cutoff2 :
        log10-transformed cell cutoff for highest cutoff (genes
        expressed in at least this amount of cells will be included).
        If None, this will default to adata.shape[0] * 0.05.

    Returns
    -------
    adata :
        filtered adata object with single cell / nucleus data
    """
    # calculate the mean of each gene across non-zero cells
    adata.var["n_cells"] = np.array((adata.X > 0).sum(0)).flatten()
    adata.var["nonz_mean"] = np.array(adata.X.sum(0)).flatten() / adata.var["n_cells"]

    if cell_count_cutoff2 is None:
        cell_count_cutoff2 = np.log10(adata.shape[0] * 0.05)

    return adata[
        :,
        (np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff2))
        | (
            np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff)
            & np.array(np.log10(adata.var["nonz_mean"]) > nonz_mean_cutoff)
        ),
    ]
