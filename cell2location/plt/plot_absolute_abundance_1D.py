import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

def plot_absolute_abundances_1D(adata_sp, subset = None, saving = False,
                               scaling = 0.15, power = 1, pws = [0,0,100,500,1000,3000,6000],
                               dimName = 'VCDepth', xlab = 'Cortical Depth', colourCode = None): 
    r""" Plot absolute abundance of celltypes in a dotplot across 1 dimension

    :param adata_sp: anndata object for spatial data with celltype abundance included in .obs (this is returned by running cell2location first)
    :param subset: optionally a boolean for only using part of the data in adata_sp
    :param saving: optionally a string value, which will result in the plot to be saved under this name
    :param scaling: how dot size should scale linearly with abundance values, default 0.15
    :param power: how dot size should scale non-linearly with abundance values, default 1 (no non-linear scaling)
    :param pws: which abundance values to show in the legend
    :param dimName: the name of the dimensions in adata_sp.obs to use for plotting
    :param xlab: the x-axis label for the plot
    :param colourCode: optionally a dictionary mapping cell type names to colours
    """ 
    
    SMALL_SIZE = 18
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    celltypes = [x.split('mean_spot_factors')[-1] for x in adata_sp.obs.columns if len(x.split('mean_spot_factors')) == 2 ]
    abundances = adata_sp.obs.loc[:,[len(x.split('mean_spot_factors')) == 2 for x in adata_sp.obs.columns]]
    
    if subset:
        celltypesForPlot = np.repeat(celltypes,sum(subset))
        vcForPlot = np.array([adata_sp.obs[dimName].loc[subset] for j in range(len(celltypes))]).flatten()
        countsForPlot = np.array([abundances.iloc[:,j].loc[subset] for j in range(len(celltypes))]) 
    else:
        celltypesForPlot = np.repeat(celltypes,np.shape(adata_sp)[0])
        vcForPlot = np.array([adata_sp.obs[dimName] for j in range(len(celltypes))]).flatten()
        countsForPlot = np.array([abundances.iloc[:,j] for j in range(len(celltypes))]) 
    
    if type(colourCode) is dict:
        colourCode = pd.DataFrame(data = colourCode.values(), index = colourCode.keys(), columns = ['Colours'])
    else:
        colourCode = pd.DataFrame(data = 'black', index = celltypes, columns = ['Colours'])
    
    coloursForPlot = np.array(colourCode.loc[np.array((celltypesForPlot)),'Colours'])
    
    plt.figure(figsize = (12,8))
    plt.scatter(vcForPlot, celltypesForPlot, s=((1-np.amin(countsForPlot*scaling) + countsForPlot*scaling))**power,
                c= coloursForPlot)

    plt.xlabel(xlab)

    # make a legend:
    for pw in pws:
        plt.scatter([], [], s=((1-np.amin(countsForPlot*scaling) + pw*scaling))**power, c="black",label=str(pw))

    h, l = plt.gca().get_legend_handles_labels()
    lgd = plt.legend(h[1:], l[1:], labelspacing=1.2, title="Total Number", borderpad=1, 
                frameon=True, framealpha=0.6, edgecolor="k", facecolor="w", bbox_to_anchor=(1.55, 0.5))
    plt.tight_layout()
    
    if saving:
        plt.savefig(saving)