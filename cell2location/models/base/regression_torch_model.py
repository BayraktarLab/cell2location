# -*- coding: utf-8 -*-
"""RegressionTorchModel Base class for model with no cell specific parameters"""

import matplotlib.pyplot as plt

# +
import numpy as np
import pandas as pd

from cell2location.models.base.torch_model import TorchModel


class RegressionTorchModel(TorchModel):
    r"""Base class for regression models with no cell-specific parameters (enable minibatch training).

    :param sample_id: str with column name in cell2covar that denotes sample
    :param cell2covar: pd.DataFrame with covariates in columns and cells in rows, rows should be named.
    :param X_data: Numpy array of gene expression (cols) in cells (rows)
    :param n_iter: number of iterations, when using minibatch, the number of epochs (passes through all data),
      supersedes self.n_iter
    :param (data_type, learning_rate, total_grad_norm_constraint, verbose, var_names, var_names_read, obs_names, fact_names):
      arguments for parent class :func:`~cell2location.models.BaseModel`
    :param minibatch_size: if None all data is used for training,
      if not None - the number of cells / observations per batch. For best results use 1024 cells per batch.
    :param minibatch_seed: order of cells in minibatch is chose randomly, so a seed for each traning restart
      should be provided
    :param prior_eps: numerical stability constant added to initial values
    :param nb_param_conversion_eps: NB distribution numerical stability constant, see :func:`~cell2location.models.TorchModel.nb_log_prob`
    :param use_cuda: boolean, telling pytorch to use the GPU (if true).
    :param use_average_as_initial_value: boolean, use average gene expression for each categorical covariate as initial value?
    :param stratify_cv: when using cross-validation on cells (selected in the training method), this is a pd.Series that
      tells :func:`~sklearn.model_selection.train_test_split` how to stratify when creating a split.
    """

    def __init__(
        self,
        sample_id,
        cell2covar: pd.DataFrame,
        X_data: np.ndarray,
        data_type="float32",
        n_iter=200000,
        learning_rate=0.001,
        total_grad_norm_constraint=200,
        verbose=True,
        var_names=None,
        var_names_read=None,
        obs_names=None,
        fact_names=None,
        minibatch_size=None,
        minibatch_seed=[41, 56, 345],
        prior_eps=1e-8,
        nb_param_conversion_eps=1e-8,
        use_cuda=False,
        use_average_as_initial_value=True,
        stratify_cv=None,
    ):

        ############# Initialise parameters ################
        # convert covariates to binary matrix
        # test for column types, get dummies for categorical / character, and just copy over continous
        cell2covar_df = pd.get_dummies(cell2covar.loc[:, ~cell2covar.columns.isin([sample_id])])
        cell2sample_df = pd.get_dummies(cell2covar[[sample_id]])
        cell2sample_covar_df = pd.concat([cell2sample_df, cell2covar_df], axis=1)

        fact_names = cell2sample_covar_df.columns
        n_fact = cell2sample_covar_df.shape[1]

        # extract obs names and sample id
        obs_names = cell2covar.index
        sample_id = cell2covar[sample_id]

        super().__init__(
            X_data,
            n_fact,
            data_type,
            n_iter,
            learning_rate,
            total_grad_norm_constraint,
            verbose,
            var_names,
            var_names_read,
            obs_names,
            fact_names,
            sample_id,
            use_cuda,
        )

        self.nb_param_conversion_eps = nb_param_conversion_eps

        self.cell_factors_df = None
        self.minibatch_size = minibatch_size
        self.minibatch_seed = minibatch_seed
        self.n_cells_total = self.n_obs
        self.which_sample = self.fact_names.isin(cell2sample_df.columns)
        self.n_samples = np.sum(self.which_sample)
        self.n_covar = self.n_fact - self.n_samples

        self.prior_eps = prior_eps

        self.cell2sample_df = cell2sample_df
        self.cell2sample_covar_df = cell2sample_covar_df
        # convert to np.ndarray
        self.cell2sample_mat = cell2sample_df.values
        self.cell2sample_covar_mat = cell2sample_covar_df.values

        # find mean and variance for each gene
        self.gene_mean = (self.X_data + self.prior_eps).mean(0).astype(self.data_type).reshape((1, self.n_var))
        self.noise_gene_mean = (self.gene_mean / 10).astype(self.data_type).reshape((1, self.n_var))
        self.prior_gene_mean = np.concatenate([self.noise_gene_mean, self.gene_mean], axis=0)

        self.stratify_cv = stratify_cv

        self.extra_data["cell2sample_covar"] = self.cell2sample_covar_mat

        if use_average_as_initial_value:
            # compute initial value for parameters: cluster averages
            self.cell2sample_covar_sig_mat = self.cell2sample_covar_mat / self.cell2sample_covar_mat.sum(0)
            self.clust_average_mat = np.dot(self.cell2sample_covar_sig_mat.T, self.X_data) + self.prior_eps
            self.clust_average_mat[self.which_sample, :] = self.clust_average_mat[self.which_sample, :] / 10

            # aver = get_cluster_averages(adata_snrna_raw, 'annotation_1') + self.prior_eps
            # variances = get_cluster_variances(adata_snrna_raw, 'annotation_1') + self.prior_eps
            # shape = aver ** 2 / variances
            # shape = shape.mean(1).values
            # overdisp_mean = shape.reshape((1, adata_snrna_raw.shape[1]))
            self.gene_E_mat = None  # np.sqrt(1 / overdisp_mean) # get gene_E ~ Exponential()
        else:
            self.clust_average_mat = None
            self.gene_E_mat = None

    # =====================Other functions======================= #
    def plot_gene_budget(self):

        plt.hist(np.log10(self.samples["post_sample_means"]["gene_level"][:, 0]), bins=50)
        plt.xlabel("Gene expression level (hierarchical)")
        plt.title("Gene expression level (hierarchical)")
        plt.tight_layout()

    def sample2df(self, gene_node_name="gene_factors", sample_type="means"):
        r"""Export cell factors as Pandas data frames.

        :param node_name: name of the cell factor model parameter to be exported
        :param gene_node_name: name of the gene factor model parameter to be exported
        :param sample_type: type of posterior sample (means, q05, q95, sds)
        :return: 8 Pandas dataframes added to model object:
                 .covariate_effects, .covariate_effects_sd, .covariate_effects_q05, .covariate_effects_q95
                 .sample_effects, .sample_effects_sd, .sample_effects_q05, .sample_effects_q95
        """

        # export parameters for covariate effects
        cov_ind = ~self.which_sample
        self.covariate_effects = pd.DataFrame.from_records(
            self.samples["post_sample_" + sample_type][gene_node_name][cov_ind, :].T,
            index=self.var_names,
            columns=[sample_type + "_cov_effect_" + i for i in self.fact_names[cov_ind]],
        )

        # export parameters for sample effects
        sample_ind = self.which_sample
        self.sample_effects = pd.DataFrame.from_records(
            self.samples["post_sample_" + sample_type][gene_node_name][sample_ind, :].T,
            index=self.var_names,
            columns=[sample_type + "_sample_effect" + i for i in self.fact_names[sample_ind]],
        )

    def annotate_cell_adata(self, adata, use_raw=True):
        r"""Add covariate and sample coefficients to anndata.var

        :param adata: anndata object to annotate
        :return: updated anndata object
        """

        if self.cell_factors_df is None:
            self.sample2df()

        if use_raw is True:

            var_index = adata.raw.var.index

            ### Covariate effect
            # add gene factors to adata
            adata.raw.var[self.covariate_effects.columns] = self.covariate_effects.loc[var_index, :]

            ### Sample effects
            # add gene factors to adata
            adata.raw.var[self.sample_effects.columns] = self.sample_effects.loc[var_index, :]

        else:

            var_index = adata.var.index

            ### Covariate effect
            # add gene factors to adata
            adata.var[self.covariate_effects.columns] = self.covariate_effects.loc[var_index, :]

            ### Sample effects
            # add gene factors to adata
            adata.var[self.sample_effects.columns] = self.sample_effects.loc[var_index, :]

        return adata
