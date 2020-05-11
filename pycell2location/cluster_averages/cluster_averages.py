### Build cell state signature matrix ###
import sys, ast, os
import anndata
import pandas as pd
import numpy as np

def get_cluster_averages(adata_ref, cluster_col):
    '''
    Parameters
    ----------
    adata_ref: anndata
        AnnData object of reference single-cell dataset
    cluster_col: str
        name of adata_ref.obs column containing cluster labels   
    
    Returns:
    --------
    pd.DataFrame of cluster average transcriptomes
    '''
    if not adata_ref.raw:
        raise ValueError("AnnData object has no raw data")
    if sum(adata_ref.obs.columns==cluster_col) != 1:
        raise ValueError("cluster_col is absent in adata_ref.obs or not unique")
        
    all_clusters = np.unique(adata_ref.obs[cluster_col])
    averages_mat = np.zeros((1, adata_ref.raw.X.shape[1]))
    
    for c in all_clusters:
        aver = adata_ref.raw.X[np.isin(adata_ref.obs[cluster_col], c),:].mean(0)
        averages_mat = np.concatenate((averages_mat, aver))
    averages_mat = averages_mat[1:, :].T
    averages_df = pd.DataFrame(data=averages_mat, 
                               index=adata_ref.raw.var_names,
                               columns=all_clusters)
    
    return(averages_df)
    
def get_cluster_averages_df(X, cluster_col):
    '''
    Parameters
    ----------
    X: pd.DataFrame
        DataFrame with spots / cells in rows and expression dimensions in columns
    cluster_col: pd.Series
        pd.Series object containing cluster labels   
    
    Returns:
    --------
    pd.DataFrame of cluster average transcriptomes
    '''
        
    all_clusters = np.unique(cluster_col)
    averages_mat = np.zeros((1, X.shape[1]))
    
    for c in all_clusters:
        aver = X.loc[np.isin(cluster_col, c),:].values.mean(0)
        averages_mat = np.concatenate((averages_mat, aver.reshape((1, X.shape[1]))))
    averages_mat = averages_mat[1:, :].T
    averages_df = pd.DataFrame(data=averages_mat, 
                               index=X.columns,
                               columns=all_clusters)
    
    return(averages_df)
