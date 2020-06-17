# +
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance


def heatmap(M, ticks=False, log=False, figsize=None, equal=False,
            row_labels=None, col_labels=None, cbar=True, cmap='RdPu', title=''):
    r""" Plot heatmap with row and column labels using plt.imshow

    :param M: np.ndarray to be visualised, or an object that can be coerced to np.ndarray
    :param ticks: boolean, show x and y axis ticks?
    :param log: boolean, color on logscale?
    :param figsize: figure size as a tuple (x, y)
    :param equal: boolean, each tile should be square (equal aspect)
    :param row_labels: names of rows (pd.Series, pd.Index or list)
    :param col_labels: names of columns (pd.Series, pd.Index or list)
    :param cbar: boolean, show colorbar?
    :param cmap: valid matplotlib colormap name
    :param title: title of the plot
    """
    if figsize is not None:
        plt.figure(figsize=figsize)

    M = np.array(M)
    if log:
        plt.imshow(M, interpolation='nearest', cmap=cmap,
                   norm=matplotlib.colors.LogNorm())
    else:
        plt.imshow(M, interpolation='nearest', cmap=cmap)

    if cbar == True:
        plt.colorbar()

    if ticks:
        plt.xticks(range(M.shape[1]))
        plt.yticks(range(M.shape[0]))
        plt.xlim(-0.5, M.shape[1] - 0.5)
        plt.ylim(M.shape[0] - 0.5, -0.5)

    if equal:
        plt.gca().set_aspect('equal', adjustable='box')

    if row_labels is not None:
        plt.yticks(range(M.shape[0]), row_labels)
        plt.ylim(M.shape[0] - 0.5, -0.5)

    if col_labels is not None:
        plt.xticks(range(M.shape[1]), col_labels, rotation=-90)
        plt.xlim(M.shape[1] - 0.5, -0.5)

    plt.title(title)

    plt.show()


def clustermap(df, cluster_rows=True, cluster_cols=True,
               figure_size=(5, 5), cmap="RdPu", log=False,
               return_linkage=False, equal=True, title=''):
    r"""Plot heatmap with hierarchically clustered rows and columns using `.heatmap`

    :param df: pandas.DataFrame to be visualised as a heatmap
    :param cluster_rows: cluster rows or keep the same order as df?
    :param cluster_cols: cluster columns or keep the same order as df?
    :param figure_size: tuple specifying figure dimensions, passed to .heatmap
    :param cmap: pyplot colormap, passed to .heatmap
    :param log: boolean, color on logscale?
    :param return_linkage: return the plot or the plot + linkage for rows and columns? If true returns a dictionary with 'plot', 'row_linkage' and 'col_linkage' elements.
    :param equal: boolean, each tile should be square (equal aspect)
    :param title: clustermap title
    """

    if cluster_rows:
        # hierarchically cluster rows
        cor_f1 = np.corrcoef(df)
        row_linkage = hierarchy.linkage(distance.pdist(cor_f1), method='average')
        row_ord = hierarchy.leaves_list(row_linkage)
    else:
        row_linkage = None
        row_ord = np.arange(df.shape[0])

    if cluster_cols:
        # hierarchically cluster columns
        cor_f1 = np.corrcoef(df.T)
        col_linkage = hierarchy.linkage(distance.pdist(cor_f1), method='average')
        col_ord = hierarchy.leaves_list(col_linkage)
    else:
        col_linkage = None
        col_ord = np.arange(df.shape[1])

    df = df.iloc[row_ord, col_ord]

    # plot heatmap
    heatmap(df.values, ticks=False, log=log, figsize=figure_size,
            equal=equal, cmap=cmap,
            row_labels=df.index, col_labels=df.columns, cbar=True, title=title)

    if return_linkage:
        return {'row_linkage': row_linkage, 'col_linkage': col_linkage,
                'row_ord': row_ord, 'col_ord': col_ord}
