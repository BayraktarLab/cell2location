# -*- coding: utf-8 -*-
"""sklearn NMF Cell neighbourhood model - de-novo factorisation of cell type density using sklearn NMF."""

# +
import sys, ast, os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pycell2location.models.base_model import BaseModel 

# defining the model itself
class CellNeighbourhood_sklearnNMF(BaseModel):
    r""" sklearn NMF Cell neighbourhood model - de-novo factorisation of cell type density using sklearn NMF.
    :param n_fact: Maximum number of cell circuits
    :param X_data: Numpy array of gene expression (cols) in cells (rows)
    :param learning_rate: ADAM learning rate for optimising Variational inference objective
    :param n_iter: number of training iterations
    :param total_grad_norm_constraint: gradient constraints in optimisation
    :param init, random_state, alpha, l1_ratio: arguments for sklearn.decomposition.NMF with sensible defaults
                            see help(sklearn.decomposition.NMF) for more details
    :param nmf_kwd_args: dictionary with more keyword arguments for sklearn.decomposition.NMF
    """

    def __init__(
        self,
        n_fact: int,
        X_data: np.ndarray,
        n_iter = 10000,
        verbose = True,
        var_names=None, var_names_read=None,
        obs_names=None, fact_names=None, sample_id=None,
        init='random', random_state=0, alpha=0.1, l1_ratio=0.5,
        nmf_kwd_args={}
    ):

        ############# Initialise parameters ################
        super().__init__(X_data, n_fact,
                         0, n_iter, 
                         0, 0,
                         verbose, var_names, var_names_read,
                         obs_names, fact_names, sample_id)
        
        self.location_factors_df = None
        self.X_data_sample = None
        
        self.init = init
        self.random_state = random_state
        np.random.seed(random_state)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.nmf_kwd_args = nmf_kwd_args
        
    def fit(self, n=3, n_type='restart'):
        r""" Find parameters using sklearn.decomposition.NMF, optinally restart several times, 
                and export parameters to self.samples['post_sample_means']
        :param n: number of independent initialisations
        :param n_type: type of repeated initialisation: 
                                  'restart' to pick different initial value,
                                  'cv' for molecular cross-validation - splits counts into n datasets, 
                                         for now, only n=2 is implemented
                                  'bootstrap' for fitting the model to multiple downsampled datasets. 
                                         Run `mod.bootstrap_data()` to generate variants of data
        :param n_iter: number of iterations, supersedes self.n_iter
        :return: exported parameters in self.samples['post_sample_means'] 
        """

        self.models = {}
        self.results = {}
        self.samples = {}
        
        self.n_type=n_type
        
        if np.isin(n_type, ['bootstrap']):
            if self.X_data_sample is None:
                self.bootstrap_data(n=n)
        elif np.isin(n_type, ['cv']):
            if self.X_data_sample is None:
                self.generate_cv_data() # cv data added to self.X_data_sample
            
        init_names = ['init_' + str(i+1) for i in np.arange(n)]

        for i, name in enumerate(init_names):
            
            # when type is molecular cross-validation or bootstrap, 
            # replace self.x_data with new data
            if np.isin(n_type, ['cv', 'bootstrap']):
                self.x_data = self.X_data_sample[i]
            else:
                self.x_data = self.X_data
            
            from sklearn.decomposition import NMF
            self.models[name] = NMF(n_components=self.n_fact, init=self.init, 
                                    alpha=self.alpha, l1_ratio=self.l1_ratio,
                                    max_iter=self.n_iter, **self.nmf_kwd_args)
            W = self.models[name].fit_transform(self.x_data)
            H = self.models[name].components_
            self.results[name] = {'post_sample_means': {'location_factors': W, 
                                              'cell_type_factors': H.T,
                                              'nUMI_factors': (W * (H.T).sum(0))},
                                  'post_sample_sds': None, 
                                  'post_sample_q05': None, 'post_sample_q95': None}
            self.samples = self.results[name]
            
            # plot training history
            if self.verbose:
                print(name + ' - iterations until convergence: ' + str(self.models[name].n_iter_));

    def evaluate_stability(self, node_name, align=True):
        r""" Evaluate stability of the solution between training initialisations
        (takes samples and correlates the values of factors between training initialisations)
        :param node_name: name of the parameter to evaluate, see `self.samples['post_sample_means'].keys()`  
                        Factors should be in columns.
        :param align: boolean, match factors between training restarts using linear_sum_assignment?
        :return: plots comparing all training initialisations to initialisation 1. 
        """

        for i in range(len(self.results.keys()) - 1):
            print(self.align_plot_stability(self.results['init_' + str(1)]['post_sample_means'][node_name],
                                            self.results['init_' + str(i+2)]['post_sample_means'][node_name],
                                            str(1), str(i+2), align=align))
    
    def compute_expected(self):
        r""" Compute expected abundance of each cell type in each location.
        """
        
        # compute the poisson rate
        self.mu = np.dot(self.samples['post_sample_means']['location_factors'],
                         self.samples['post_sample_means']['cell_type_factors'].T) 
        
    def compute_expected_fact(self, fact_ind=None):
        r""" Compute expected abundance of each cell type in each location
        that comes from a subset of factors. E.g. expressed factors in self.fact_filt
        """
        
        if fact_ind is None:
            fact_ind = self.fact_filt
        
        # compute the poisson rate
        self.mu = np.dot(self.samples['post_sample_means']['location_factors'][:,fact_ind],
                         self.samples['post_sample_means']['cell_type_factors'].T[fact_ind,:])
        
    def plot_posterior_mu_vs_data(self, mu_node_name='mu', data_node='X_data'):
        r""" Plot expected value of the model (e.g. mean of poisson distribution)
        :param mu_node_name: name of the object slot containing expected value
        :param data_node: name of the object slot containing data
        """
        
        if type(mu_node_name) is str:
            mu = getattr(self, mu_node_name)
        else:
            mu = mu_node_name
        
        if type(data_node) is str:
            data_node = getattr(self, data_node)
        
        plt.hist2d(data_node.flatten(),
                   mu.flatten(),
                   bins = 50, norm=matplotlib.colors.LogNorm())
        plt.xlabel('Data, values')
        plt.ylabel('Posterior sample, values')
        plt.title('UMI counts (all spots, all genes)')
        plt.tight_layout()
        
    def sample2df(self, node_name='nUMI_factors',
                  gene_node_name = 'cell_type_factors'):
        r""" Export cell factors as Pandas data frames.
        :param node_name: name of the cell factor model parameter to be exported
        :param gene_node_name: name of the gene factor model parameter to be exported
        :return: 8 Pandas dataframes added to model object:
                 .cell_factors_df, .cell_factors_sd, .cell_factors_q05, .cell_factors_q95
                 .gene_loadings, .gene_loadings_sd, .gene_loadings_q05, .gene_loadings_q95
        """
        
        # export location factors
        self.location_factors_df = \
            pd.DataFrame.from_records(self.samples['post_sample_means'][node_name],
                                      index=self.obs_names,
                                      columns=['mean_' + node_name + i for i in self.fact_names])
        
        #self.location_factors_sd = \
        #    pd.DataFrame.from_records(self.samples['post_sample_sds'][node_name],
        #                              index=self.obs_names,
        #                            columns=['sd_' + node_name + i for i in self.fact_names])
        
        #self.location_factors_q05 = \
        #    pd.DataFrame.from_records(self.samples['post_sample_q05'][node_name],
        #                              index=self.obs_names,
        #                            columns=['q05_' + node_name + i for i in self.fact_names])
        #self.location_factors_q95 = \
        #    pd.DataFrame.from_records(self.samples['post_sample_q95'][node_name],
        #                              index=self.obs_names,
        #                            columns=['q95_' + node_name + i for i in self.fact_names])
        
        # export cell type factors
        self.cell_type_loadings = \
        pd.DataFrame.from_records(self.samples['post_sample_means'][gene_node_name],
                                  index=self.var_names,
                                  columns=['mean_' + gene_node_name + i for i in self.fact_names])
        
        self.cell_type_fractions = (self.cell_type_loadings.T / self.cell_type_loadings.sum(1)).T 
        
        self.cell_type_loadings_sd = None #\
        #pd.DataFrame.from_records(self.samples['post_sample_sds'][gene_node_name],
        #                          index=self.var_names,
        #                          columns=['sd_' + gene_node_name + i for i in self.fact_names])
        
        self.cell_type_loadings_q05 = None #\
        #pd.DataFrame.from_records(self.samples['post_sample_q05'][gene_node_name],
        #                          index=self.var_names,
        #                          columns=['q05_' + gene_node_name + i for i in self.fact_names])
        
        self.cell_type_loadings_q95 = None #\
        #pd.DataFrame.from_records(self.samples['post_sample_q95'][gene_node_name],
        #                          index=self.var_names,
        #                          columns=['q95_' + gene_node_name + i for i in self.fact_names])
        
        
    def annotate_adata(self, adata):
        r""" Add location factors to anndata.obs
        :param adata: anndata object to annotate
        :return: updated anndata object
        """
        
        if self.location_factors_df is None:
            self.sample2df()
            
        ### location factors
        # add location factors to adata
        adata.obs[self.location_factors_df.columns] = self.location_factors_df.loc[adata.obs.index,:]

        # add location factor sd to adata
        #adata.obs[self.location_factors_sd.columns] = self.location_factors_sd.loc[adata.obs.index,:]

        # add location factor 5% and 95% quantile to adata
        #adata.obs[self.location_factors_q05.columns] = self.location_factors_q05.loc[adata.obs.index,:]
        #adata.obs[self.location_factors_q95.columns] = self.location_factors_q95.loc[adata.obs.index,:]
        
        ### Cell type factors
        # add gene factors to adata
        #adata.var[self.cell_type_loadings.columns] = self.cell_type_loadings.loc[adata.var.index,:]

        # add gene factor sd to adata
        #adata.var[self.cell_type_loadings_sd.columns] = self.cell_type_loadings_sd.loc[adata.var.index,:]

        # add gene factor 5% and 95% quantile to adata
        #adata.var[self.cell_type_loadings_q05.columns] = self.cell_type_loadings_q05.loc[adata.var.index,:]
        #adata.var[self.cell_type_loadings_q95.columns] = self.cell_type_loadings_q95.loc[adata.var.index,:]
        
        return(adata)
