# -*- coding: utf-8 -*-
"""Base spot location class"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from cell2location.models.pyro.pyro_model import PyroModel


# base model class - defining shared methods but not the model itself
class PyroLocModel(PyroModel):
    r"""Base class for pyro supervised location models.

    :param cell_state_mat: Pandas data frame with gene signatures - genes in row, cell states or factors in columns
    :param X_data: Numpy array of gene expression (cols) in spatial locations (rows)
    :param learning_rate: ADAM learning rate for optimising Variational inference objective
    :param n_iter: number of training iterations
    :param total_grad_norm_constraint: gradient constraints in optimisation
    """

    def __init__(
            self,
            cell_state_mat: np.ndarray,
            X_data: np.ndarray,
            data_type: str = 'float32',
            n_iter=20000,
            learning_rate=0.005,
            total_grad_norm_constraint=200,
            use_cuda=False,
            verbose=True,
            var_names=None, var_names_read=None,
            obs_names=None, fact_names=None, sample_id=None,
            minibatch_size=None,
            minibatch_seed=42
    ):

        ############# Initialise parameters ################
        super().__init__(X_data=X_data, n_fact=cell_state_mat.shape[1],
                         data_type=data_type, n_iter=n_iter,
                         learning_rate=learning_rate, total_grad_norm_constraint=total_grad_norm_constraint,
                         use_cuda=use_cuda, verbose=verbose, var_names=var_names, var_names_read=var_names_read,
                         obs_names=obs_names, fact_names=fact_names, sample_id=sample_id,
                         minibatch_size=minibatch_size, minibatch_seed=minibatch_seed)

        self.cell_state_mat = cell_state_mat
        # Pass data to pyro / pytorch
        self.cell_state = torch.tensor(cell_state_mat.astype(self.data_type))  # .double()
        if self.use_cuda:
            # move tensors and modules to CUDA
            self.cell_state = self.cell_state.cuda()

    def evaluate_stability(self, n_samples=1000, align=False):
        r""" Evaluate stability in factor contributions to spots.
        """

        self.b_evaluate_stability(node='spot_factors', n_samples=n_samples, align=align, transpose=False)

    def sample2df(self, node_name='nUMI_factors'):
        r""" Export spot factors as Pandas data frames.

        :param node_name: name of the model parameter to be exported
        :return: 4 Pandas dataframes added to model object:
            .spot_factors_df, .spot_factors_sd, .spot_factors_q05, .spot_factors_q95
        """

        if len(self.samples) == 0:
            raise ValueError(
                'Please run `.sample_posterior()` first to generate samples & summarise posterior of each parameter')

        self.spot_factors_df = \
            pd.DataFrame.from_records(self.samples['post_sample_means'][node_name],
                                      index=self.obs_names,
                                      columns=['mean_' + node_name + i for i in self.fact_names])

        self.spot_factors_sd = \
            pd.DataFrame.from_records(self.samples['post_sample_sds'][node_name],
                                      index=self.obs_names,
                                      columns=['sd_' + node_name + i for i in self.fact_names])

        self.spot_factors_q05 = \
            pd.DataFrame.from_records(self.samples['post_sample_q05'][node_name],
                                      index=self.obs_names,
                                      columns=['q05_' + node_name + i for i in self.fact_names])

        self.spot_factors_q95 = \
            pd.DataFrame.from_records(self.samples['post_sample_q95'][node_name],
                                      index=self.obs_names,
                                      columns=['q95_' + node_name + i for i in self.fact_names])

    def annotate_spot_adata(self, adata):
        r""" Add spot factors to anndata.obs

        :param adata: anndata object to annotate
        :return: updated anndata object
        """

        if self.spot_factors_df is None:
            self.sample2df()

        # add cell factors to adata
        adata.obs[self.spot_factors_df.columns] = self.spot_factors_df.loc[adata.obs.index, :]

        # add cell factor sd to adata
        adata.obs[self.spot_factors_sd.columns] = self.spot_factors_sd.loc[adata.obs.index, :]

        # add cell factor 5% and 95% quantiles to adata
        adata.obs[self.spot_factors_q05.columns] = self.spot_factors_q05.loc[adata.obs.index, :]
        adata.obs[self.spot_factors_q95.columns] = self.spot_factors_q95.loc[adata.obs.index, :]

        return (adata)

    def plot_biol_spot_nUMI(self, fact_name='nUMI_factors'):
        plt.hist(np.log10(self.samples['post_sample_means'][fact_name].sum(1)), bins=50)
        plt.xlabel('Biological spot nUMI (log10)')
        plt.title('Biological spot nUMI')
        plt.tight_layout()

    def plot_spot_add(self):
        plt.hist(np.array(np.log10(self.samples['post_sample_means']['spot_add']).flatten()), bins=50)
        plt.xlabel('UMI unexplained by biological factors')
        plt.title('Additive technical spot nUMI')
        plt.tight_layout()

    def plot_gene_E(self):
        plt.hist((np.array(self.samples['post_sample_means']['gene_E']).flatten()), bins=50)
        plt.xlabel('E_g overdispersion parameter')
        plt.title('E_g overdispersion parameter')
        plt.tight_layout()

    def plot_gene_add(self):
        plt.hist((np.array(self.samples['post_sample_means']['gene_add']).flatten()), bins=50)
        plt.xlabel('S_g additive background noise parameter')
        plt.title('S_g additive background noise parameter')
        plt.tight_layout()

    def plot_gene_level(self):
        plt.hist((np.array(self.samples['post_sample_means']['gene_level']).flatten()), bins=50)
        plt.xlabel('M_g expression level scaling parameter')
        plt.title('M_g expression level scaling parameter')
        plt.tight_layout()


    def compute_expected(self):
        r"""Compute expected expression of each gene in each spot (Poisson mu). Useful for evaluating how well
            the model learned expression pattern of all genes in the data.
        """

        # compute the poisson rate
        self.mu = np.dot(self.samples['post_sample_means']['spot_factors'],
                         self.samples['post_sample_means']['gene_factors'].T) \
                  * self.samples['post_sample_means']['gene_level'].T \
                  + self.samples['post_sample_means']['gene_add'].T \
                  + self.samples['post_sample_means']['spot_add']

    def compute_expected_clust(self, fact_ind):
        r""" Compute expected expression of each gene in each spot that comes from one reference cluster (Poisson mu).
        """

        # compute the poisson rate
        self.mu = np.dot(self.samples['post_sample_means']['spot_factors'][:, fact_ind],
                         self.samples['post_sample_means']['gene_factors'].T[fact_ind, :]) \
                  * self.samples['post_sample_means']['gene_level'].T
