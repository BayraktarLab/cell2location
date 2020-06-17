# -*- coding: utf-8 -*-
"""LocationModelNB4V7_V4_V4 Cell location model with E_g overdispersion & NB likelihood 
    - similar to LocationModelV7_V4_V4"""

import sys, ast, os
import time
import itertools
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import theano.tensor as tt
import pymc3 as pm
import pickle
import theano

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os

from pycell2location.models.pymc3_loc_model import Pymc3LocModel 

# defining the model itself
class LocationModelNB4V7_V4_V4(Pymc3LocModel):
    r"""LocationModelNB4V7_V4_V4 Cell location model with E_g overdispersion & NB likelihood
         - similar to LocationModelNB2V7_V4_V4
         pymc3 NB parametrisation but overdisp priors as described here https://statmodeling.stat.columbia.edu/2018/04/03/justify-my-love/
    :param cell_state_mat: Pandas data frame with gene signatures - genes in row, cell states or factors in columns
    :param X_data: Numpy array of gene expression (cols) in spatial locations (rows)
    :param learning_rate: ADAM learning rate for optimising Variational inference objective
    :param n_iter: number of training iterations
    :param total_grad_norm_constraint: gradient constraints in optimisation
    :param gene_level_prior: prior on change in sensitivity between single cell and spatial (mean), 
                                how much it varies across cells (sd),
                                and how certain we are in those numbers (mean_var_ratio) 
                                 - by default the variance in our prior of mean and sd is equal to the mean and sd
                                 descreasing this number means having higher uncertainty about your prior
    :param cell_number_prior: prior on cell density parameter:
                                cells_per_spot - what is the number of cells you expect per location?
                                factors_per_spot - what is the number of cell types 
                                                        / number of factors expressed per location?
                                cells_mean_var_ratio, factors_mean_var_ratio - uncertainty in both prior
                                                        expressed as a mean/var ratio, numbers < 1 mean high uncertainty
    :param phi_hyp_prior: prior on overdispersion parameter, rate of exponential distribution over phi / theta
    """

    def __init__(
        self,
        cell_state_mat: np.ndarray,
        X_data: np.ndarray,
        data_type: str = 'float32',
        n_iter = 200000,
        learning_rate = 0.001,
        total_grad_norm_constraint = 200,
        verbose = True,
        var_names=None, var_names_read=None,
        obs_names=None, fact_names=None, sample_id=None,
        gene_level_prior={'mean': 1/2, 'sd': 1/8, 'mean_var_ratio': 1},
        cell_number_prior={'cells_per_spot': 7, 'factors_per_spot': 6,
                           'cells_mean_var_ratio': 1, 'factors_mean_var_ratio': 1},
        phi_hyp_prior={'mean': 3, 'sd': 1}
    ):

        ############# Initialise parameters ################
        super().__init__(cell_state_mat, X_data,
                         data_type, n_iter, 
                         learning_rate, total_grad_norm_constraint,
                         verbose, var_names, var_names_read,
                         obs_names, fact_names, sample_id)

        self.gene_level_prior = gene_level_prior
        self.cell_number_prior = cell_number_prior
        self.phi_hyp_prior = phi_hyp_prior
        
        ############# Define the model ################
        self.model = pm.Model()

        with self.model:
            
            # =====================Gene expression level scaling======================= #
            # Explains difference in expression between genes and 
            # how it differs in single cell and spatial technology
            # compute hyperparameters from mean and sd
            shape = gene_level_prior['mean']**2 / gene_level_prior['sd']**2
            rate = gene_level_prior['mean'] / gene_level_prior['sd']**2 
            shape_var = shape / gene_level_prior['mean_var_ratio']
            rate_var = rate / gene_level_prior['mean_var_ratio']
            self.gene_level_alpha_hyp = pm.Gamma('gene_level_alpha_hyp',
                                                 mu=shape, sigma=np.sqrt(shape_var),
                                                 shape=(1, 1))
            self.gene_level_beta_hyp = pm.Gamma('gene_level_beta_hyp', 
                                                 mu=rate, sigma=np.sqrt(rate_var),
                                                 shape=(1, 1))
        
            self.gene_level = pm.Gamma('gene_level', self.gene_level_alpha_hyp,
                                       self.gene_level_beta_hyp, shape=(self.n_genes, 1))
        
            # scale cell state factors by gene_level
            self.gene_factors = pm.Deterministic('gene_factors', self.cell_state)
            #tt.printing.Print('gene_factors sum')(gene_factors.sum(0).shape)
            #tt.printing.Print('gene_factors sum')(gene_factors.sum(0))
    
            # =====================Spot factors======================= #
            # prior on spot factors reflects the number of cells, fraction of their cytoplasm captured, 
            # times heterogeniety in the total number of mRNA between individual cells with each cell type
            self.cells_per_spot = pm.Gamma('cells_per_spot',
                                           mu=cell_number_prior['cells_per_spot'],
                                           sigma=np.sqrt(cell_number_prior['cells_per_spot'] \
                                                         / cell_number_prior['cells_mean_var_ratio']),
                                           shape=(self.n_cells, 1))
            self.factors_per_spot = pm.Gamma('factors_per_spot', 
                                             mu=cell_number_prior['factors_per_spot'], 
                                             sigma=np.sqrt(cell_number_prior['factors_per_spot'] \
                                                           / cell_number_prior['factors_mean_var_ratio']),
                                             shape=(self.n_cells, 1))
            
            shape = self.factors_per_spot / np.array(self.n_fact).reshape((1, 1))
            rate = tt.ones((1, 1)) / self.cells_per_spot * self.factors_per_spot
            self.spot_factors = pm.Gamma('spot_factors', alpha=shape, beta=rate,
                                         shape=(self.n_cells, self.n_fact))
    
            # =====================Spot-specific additive component======================= #
            # molecule contribution that cannot be explained by cell state signatures
            # these counts are distributed between all genes not just expressed genes
            self.spot_add_hyp = pm.Gamma('spot_add_hyp', 1, 0.1, shape=2)
            self.spot_add = pm.Gamma('spot_add', self.spot_add_hyp[0],
                                     self.spot_add_hyp[1], shape=(self.n_cells, 1))
            
            # =====================Gene-specific additive component ======================= #
            # per gene molecule contribution that cannot be explained by cell state signatures
            # these counts are distributed equally between all spots (e.g. background, free-floating RNA)
            self.gene_add_hyp = pm.Gamma('gene_add_hyp', 1, 1, shape=2)
            self.gene_add = pm.Gamma('gene_add', self.gene_add_hyp[0],
                                     self.gene_add_hyp[1], shape=(self.n_genes, 1))
            
            # =====================Gene-specific overdispersion ======================= #
            self.phi_hyp = pm.Gamma('phi_hyp', mu=phi_hyp_prior['mean'], 
                                    sigma=phi_hyp_prior['sd'], shape=(1, 1))
            self.gene_E = pm.Exponential('gene_E', self.phi_hyp, shape=(self.n_genes, 1))
    
            # =====================Expected expression ======================= #
            # expected expression
            self.mu_biol = pm.math.dot(self.spot_factors, self.gene_factors.T) * self.gene_level.T \
                                    + self.gene_add.T + self.spot_add
            #tt.printing.Print('mu_biol')(self.mu_biol.shape)
    
            # =====================DATA likelihood ======================= #
            # Likelihood (sampling distribution) of observations & add overdispersion via NegativeBinomial / Poisson
            self.data_target = pm.NegativeBinomial('data_target', mu=self.mu_biol,
                                                   alpha=1/(self.gene_E.T * self.gene_E.T),
                                          observed=self.x_data,
                                          total_size=self.X_data.shape)
                                          
            # =====================Compute nUMI from each factor in spots  ======================= #                          
            self.nUMI_factors = pm.Deterministic('nUMI_factors',
                                                 (self.spot_factors * (self.gene_factors * self.gene_level).sum(0)))
    
    def plot_posterior_vs_dataV1(self):
        self.plot_posterior_vs_data(gene_fact_name = 'gene_factors',
                               cell_fact_name = 'spot_factors_scaled')
    
    def plot_biol_spot_nUMI(self, fact_name='nUMI_factors'):
        
        plt.hist(np.log10(self.samples['post_sample_means'][fact_name].sum(1)), bins = 50)
        plt.xlabel('Biological spot nUMI (log10)')
        plt.title('Biological spot nUMI')
        plt.tight_layout()
        
    def plot_spot_add(self):
        
        plt.hist(np.log10(self.samples['post_sample_means']['spot_add'][:,0]), bins = 50)
        plt.xlabel('UMI unexplained by biological factors')
        plt.title('Additive technical spot nUMI')
        plt.tight_layout()
        
    def plot_gene_E(self):
        
        plt.hist((self.samples['post_sample_means']['gene_E'][:,0]), bins = 50)
        plt.xlabel('E_g overdispersion parameter')
        plt.title('E_g overdispersion parameter')
        plt.tight_layout()
        
    def plot_gene_add(self):
        
        plt.hist((self.samples['post_sample_means']['gene_add'][:,0]), bins = 50)
        plt.xlabel('S_g additive background noise parameter')
        plt.title('S_g additive background noise parameter')
        plt.tight_layout()
        
    def plot_gene_level(self):
        
        plt.hist((self.samples['post_sample_means']['gene_level'][:,0]), bins = 50)
        plt.xlabel('M_g expression level scaling parameter')
        plt.title('M_g expression level scaling parameter')
        plt.tight_layout()
        
    def compute_expected(self):
        r""" Compute expected expression of each gene in each spot (Poisson mu). Useful for evaluating how well the model learned expression pattern of all genes in the data.
        """
        
        # compute the poisson rate
        self.mu = (np.dot(self.samples['post_sample_means']['spot_factors'],
                         self.samples['post_sample_means']['gene_factors'].T) \
        * self.samples['post_sample_means']['gene_level'].T \
        + self.samples['post_sample_means']['gene_add'].T \
        + self.samples['post_sample_means']['spot_add']) #\
       # * self.samples['post_sample_means']['gene_E'] 
