# -*- coding: utf-8 -*-
"""RegressionNBV2Torch Negative binomial regression model with sample scaling in pytorch."""

# +
import sys, ast, os
import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib
import os


from cell2location.models.regression_torch_model import RegressionTorchModel
from cell2location.cluster_averages.cluster_averages import get_cluster_averages
from cell2location.cluster_averages.cluster_averages import get_cluster_variances

# defining the model itself
class RegressionNBV2TorchModule(nn.Module):
    r"""Module class that defines the model graph and parameters.
    :param sample_col: str with column name in cell2covar that denotes sample
    :param cell2covar: pd.DataFrame with covariates in columns and cells in rows, rows should be named.
    :param n_cells: number of cells / spots / observations
    :param n_genes: number of genes / variables
    :param n_fact: number of factors / reference signatures (extracted from `cell_state_mat.shape[1]`)
    :param cell_state_mat: reference signatures with n_genes rows and signatures in columns
    """
    
    def __init__(self, n_cells, n_genes, n_fact, clust_average_mat, 
                 which_sample, data_type = 'float32', eps=1e-8):
        super().__init__()
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_fact = n_fact
        self.data_type = data_type
        self.which_sample = which_sample
        self.n_samples = np.sum(self.which_sample)
        self.eps = eps
        self.clust_average_mat = clust_average_mat

        # ====================Covariate and sample effects======================= #
        self.gene_factors_log = nn.Parameter(torch.Tensor(self.n_fact, self.n_genes))
        
        # =====================Sample-specific scaling of expression levels ======================= #
        self.sample_scaling_log = nn.Parameter(torch.Tensor(self.n_samples, 1))
        
        # =====================Gene-specific overdispersion ======================= #
        self.gene_E_log = nn.Parameter(torch.Tensor(1, self.n_genes))

    def forward(self, cell2sample_covar):
        r""" Computes expected value of the model (mu_biol) and gene-overdispersion (gene_E).
        :return: a list with mu_biol and gene_E
        """
        
        # sample-specific scaling of expression levels
        self.cell2sample_scaling = torch.mm(cell2sample_covar[:,self.which_sample], 
                                               torch.exp(self.sample_scaling_log))
        self.mu_biol = torch.mm(cell2sample_covar, 
                                torch.exp(self.gene_factors_log)) * self.cell2sample_scaling
        
        self.gene_E = torch.tensor(1) / (torch.exp(self.gene_E_log) * torch.exp(self.gene_E_log))
        
        return [self.mu_biol, self.gene_E]
    
    def initialize_parameters(self):
        r""" Randomly initialise parameters
        """
        
        if self.clust_average_mat is not None:
            self.gene_factors_log.data = \
            nn.Parameter(torch.tensor(np.log(self.clust_average_mat.astype(self.data_type) + self.eps)))
        else:
            self.gene_factors_log.data.normal_(mean = 0, std = 1/2)
            
        self.sample_scaling_log.data.normal_(mean = 0, std = 1/20)
        self.gene_E_log.data.normal_(mean = -1/2, std = 1/2)
        
    def export_parameters(self):
        r""" Compute parameters from their real number representation.
        :return: a dictionary with parameters
        """
        
        export = {'gene_factors': torch.exp(self.gene_factors_log),
                  'sample_scaling': torch.exp(self.sample_scaling_log),
                  'gene_E': torch.exp(self.gene_E_log)
                 }
        
        return export

class RegressionNBV2Torch(RegressionTorchModel):
    r"""RegressionNBV2Torch Negative binomial regression model with sample scaling in pytorch.
    :param sample_col: str with column name in cell2covar that denotes sample
    :param cell2covar: pd.DataFrame with covariates in columns and cells in rows, rows should be named.
    :param cell_state_mat: Pandas data frame with gene signatures - genes in row, cell states or factors in columns
    :param X_data: Numpy array of gene expression (cols) in spatial locations (rows)
    :param learning_rate: ADAM learning rate for optimising Variational inference objective
    :param n_iter: number of training iterations
    :param total_grad_norm_constraint: gradient constraints in optimisation
    """

    def __init__(
        self,
        sample_id,
        cell2covar: pd.DataFrame,
        X_data: np.ndarray,
        data_type = 'float32',
        n_iter = 200000,
        learning_rate = 0.001,
        total_grad_norm_constraint = 200,
        verbose = True,
        var_names=None, var_names_read=None,
        obs_names=None, fact_names=None,
        minibatch_size=None, minibatch_seed=[41, 56, 345],
        phi_hyp_prior=None, prior_eps=1e-8,
        nb_param_conversion_eps=1e-8,
        use_cuda=False,
        use_average_as_initial_value=True,
        stratify_cv=None,
    ):

        ############# Initialise parameters ################
        super().__init__(sample_id, cell2covar, X_data,
                         data_type, n_iter, 
                         learning_rate, total_grad_norm_constraint,
                         verbose, var_names, var_names_read,
                         obs_names, fact_names, 
                         minibatch_size, minibatch_seed,
                         phi_hyp_prior, prior_eps,
                         nb_param_conversion_eps, use_cuda, 
                         use_average_as_initial_value, stratify_cv)
        
        ############# Define the model ################
        
        self.model = RegressionNBV2TorchModule(self.n_cells, self.n_genes, 
                                               self.n_fact, self.clust_average_mat,
                                               which_sample=self.which_sample,
                                               data_type=self.data_type)
        
    # =====================DATA likelihood loss function======================= #
    # define cost function 
    def loss(self, param, data, l2_weight=None, sample_scaling_weight=0.5):
        r""" Method that returns loss (a single number)
        :param param: list with [mu, theta]
        :param data: data (cells * genes)
        :param l2_weight: strength of L2 regularisation (on exported parameters), if None - no regularisation 
        """
        
        # initialise extra penalty
        l2_reg = 0
        
        if l2_weight is not None:
            
            param_dict = self.model.export_parameters()
            sample_scaling = param_dict['sample_scaling']
            param_val = [param_dict[i] for i in param_dict.keys() if i != 'sample_scaling']
            
            # regularise sample_scaling
            l2_reg = l2_reg + (sample_scaling - 1).pow(2).sum()
            
            # regularise other parameters
            for i in param_val:
                l2_reg = l2_reg + l2_weight * i.pow(2).sum()
            
        return -self.nb_log_prob(param, data).sum() + l2_reg
    
    
    
    # =====================Other functions======================= #
        
    def compute_expected(self):
        r""" Compute expected expression of each gene in each cell (Poisson mu).
        """
        
        # compute the poisson rate
        self.mu = np.dot(self.cell2sample_covar_mat,
                         self.samples['post_sample_means']['gene_factors']) \
        * np.dot(self.cell2sample_mat,
                 self.samples['post_sample_means']['sample_scaling'])
        
    def compute_expected_fact(self, fact_ind, 
                              sample_scaling=True):
        r""" Compute expected expression of each gene in each cell 
        that comes from a subset of factors (Poisson mu). E.g. expressed factors in self.fact_filt (by default)
        :param sample_scaling: if False do not rescale levels to specific sample
        """
        
        # compute the poisson rate
        self.mu = np.dot(self.cell2sample_covar_mat[:,fact_ind],
                         self.samples['post_sample_means']['gene_factors'][fact_ind,:]) 
        if sample_scaling:
            self.mu = self.mu * np.dot(self.cell2sample_mat,
                                       self.samples['post_sample_means']['sample_scaling'])
        
    def normalise_by_sample_scaling(self, remove_additive=True):
        r""" Normalise expression data by inferred sample scaling parameters.
        """
        
        corrected = self.X_data / np.dot(self.cell2sample_mat, 
                                         self.samples['post_sample_means']['sample_scaling'])
        
        if remove_additive:
            # remove additive sample effects
            corrected = corrected - np.dot(self.cell2sample_mat, reg_mod.sample_effects.values.T)
        
        corrected = corrected - corrected.min()
        
        return corrected 
