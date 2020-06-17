### Build cell state signature matrix ###
import sys, ast, os
import anndata
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# +
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
    pd.DataFrame of cluster average expression of each gene
    '''
    if not adata_ref.raw:
        raise ValueError("AnnData object has no raw data")
    if sum(adata_ref.obs.columns==cluster_col) != 1:
        raise ValueError("cluster_col is absent in adata_ref.obs or not unique")
        
    all_clusters = np.unique(adata_ref.obs[cluster_col])
    averages_mat = np.zeros((1, adata_ref.raw.X.shape[1]))
    
    for c in all_clusters:
        sparse_subset = csr_matrix(adata_ref.raw.X[np.isin(adata_ref.obs[cluster_col], c),:])
        aver = sparse_subset.mean(0)
        averages_mat = np.concatenate((averages_mat, aver))
    averages_mat = averages_mat[1:, :].T
    averages_df = pd.DataFrame(data=averages_mat, 
                               index=adata_ref.raw.var_names,
                               columns=all_clusters)
    
    return(averages_df)

def get_cluster_variances(adata_ref, cluster_col):
    '''
    Parameters
    ----------
    adata_ref: anndata
        AnnData object of reference single-cell dataset
    cluster_col: str
        name of adata_ref.obs column containing cluster labels   
    
    Returns:
    --------
    pd.DataFrame of within cluster variance of each gene
    '''
    if not adata_ref.raw:
        raise ValueError("AnnData object has no raw data")
    if sum(adata_ref.obs.columns==cluster_col) != 1:
        raise ValueError("cluster_col is absent in adata_ref.obs or not unique")
        
    all_clusters = np.unique(adata_ref.obs[cluster_col])
    var_mat = np.zeros((1, adata_ref.raw.X.shape[1]))
    
    for c in all_clusters:
        sparse_subset = csr_matrix(adata_ref.raw.X[np.isin(adata_ref.obs[cluster_col], c),:])
        c = sparse_subset.copy(); 
        c.data **= 2
        var = c.mean(0) - (np.array(sparse_subset.mean(0)) ** 2)
        del c
        var_mat = np.concatenate((var_mat, var))
    var_mat = var_mat[1:, :].T
    var_df = pd.DataFrame(data=var_mat, 
                               index=adata_ref.raw.var_names,
                               columns=all_clusters)
    
    return(var_df)

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
    pd.DataFrame of cluster average expression of each gene
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

def get_cluster_variances_df(X, cluster_col):
    '''
    Parameters
    ----------
    X: pd.DataFrame
        DataFrame with spots / cells in rows and expression dimensions in columns
    cluster_col: pd.Series
        pd.Series object containing cluster labels   
    
    Returns:
    --------
    pd.DataFrame of within cluster variances of each gene
    '''
        
    all_clusters = np.unique(cluster_col)
    averages_mat = np.zeros((1, X.shape[1]))
    
    for c in all_clusters:
        aver = X.loc[np.isin(cluster_col, c),:].values.var(0)
        averages_mat = np.concatenate((averages_mat, aver.reshape((1, X.shape[1]))))
    averages_mat = averages_mat[1:, :].T
    averages_df = pd.DataFrame(data=averages_mat, 
                               index=X.columns,
                               columns=all_clusters)
    
    return(averages_df)
