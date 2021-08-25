# -*- coding: utf-8 -*-
"""Base spot location class"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano
from tqdm.auto import tqdm

from cell2location.models.base.pymc3_model import Pymc3Model
from cell2location.plt.plot_factor_spatial import plot_factor_spatial


# base model class - defining shared methods but not the model itself
class Pymc3LocModel(Pymc3Model):
    r"""Base class for pymc3 supervised location models.

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
        data_type: str = "float32",
        n_iter=200000,
        learning_rate=0.001,
        total_grad_norm_constraint=200,
        verbose=True,
        var_names=None,
        var_names_read=None,
        obs_names=None,
        fact_names=None,
        sample_id=None,
    ):

        ############# Initialise parameters ################
        super().__init__(
            X_data,
            cell_state_mat.shape[1],
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
        )

        self.cell_state_mat = cell_state_mat
        self.spot_factors_df = None
        self.X_data_sample = None

        # Pass extra data to theano
        self.cell_state = theano.shared(cell_state_mat.astype(self.data_type))

    def evaluate_stability(self, n_samples=1000, align=True):
        r"""Evaluate stability in factor contributions to spots."""

        self.b_evaluate_stability(node=self.spot_factors, n_samples=n_samples, align=align)

    def sample2df(self, node_name="nUMI_factors"):
        r"""Export spot factors as Pandas data frames.

        :param node_name: name of the model parameter to be exported
        :return: 4 Pandas dataframes added to model object:
            .spot_factors_df, .spot_factors_sd, .spot_factors_q05, .spot_factors_q95
        """

        if len(self.samples) == 0:
            raise ValueError(
                "Please run `.sample_posterior()` first to generate samples & summarise posterior of each parameter"
            )

        self.spot_factors_df = pd.DataFrame.from_records(
            self.samples["post_sample_means"][node_name],
            index=self.obs_names,
            columns=["mean_" + node_name + i for i in self.fact_names],
        )

        self.spot_factors_sd = pd.DataFrame.from_records(
            self.samples["post_sample_sds"][node_name],
            index=self.obs_names,
            columns=["sd_" + node_name + i for i in self.fact_names],
        )

        self.spot_factors_q05 = pd.DataFrame.from_records(
            self.samples["post_sample_q05"][node_name],
            index=self.obs_names,
            columns=["q05_" + node_name + i for i in self.fact_names],
        )

        self.spot_factors_q95 = pd.DataFrame.from_records(
            self.samples["post_sample_q95"][node_name],
            index=self.obs_names,
            columns=["q95_" + node_name + i for i in self.fact_names],
        )

    def annotate_spot_adata(self, adata):
        r"""Add spot factors to anndata.obs

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

        return adata

    def plot_tracking_history(
        self,
        adata,
        plot_dir,
        sample="s144600",
        n_columns=10,
        column_ind=None,
        figure_size=(40, 10),
        point_size=0.8,
        text_size=9,
    ):
        r"""Plot tracking history of spot-specific parameters in 2D

        :param adata: anndata object that contains locations of spots
        :param plot_dir: directory where to save plots
        :param sample: string, selected sample ID
        :param n_columns: number of columns in a plot
        :param column_ind: which columns in `mod.tracking['init_1']['samples_df'][i]` to plot? Defalt 'None' corresponds to all.
        """

        from plotnine import ggtitle

        # for each training initialisation
        for n_init in self.tracking.keys():
            # for each recorded step
            for i in tqdm(range(len(self.tracking[n_init]["samples_df"]))):

                step = self.tracking[n_init]["samples_df"][i]
                step_name = "iter_" + str((i) * self.tracking_every)

                # add RMSE
                rmse_total = self.tracking[n_init]["rmse"][i]["rmse_total"]
                step_name_r = step_name + " RMSE:" + str(np.around(rmse_total, 3))
                # add RMSE on validation data
                if "rmse_total_cv" in self.tracking[n_init]["rmse"][i].keys():
                    rmse_total_cv = self.tracking[n_init]["rmse"][i]["rmse_total_cv"]
                    step_name_r = step_name_r + " RMSE validation:" + str(np.around(rmse_total_cv, 3))

                if column_ind is None:
                    column_ind = np.arange(step.shape[1])
                p = (
                    plot_factor_spatial(
                        adata=adata,
                        fact=step,
                        cluster_names=step.columns,
                        fact_ind=column_ind,
                        n_columns=n_columns,
                        sample_name=sample,
                        figure_size=figure_size,
                        point_size=point_size,
                        text_size=text_size,
                    )
                    + ggtitle(step_name_r)
                )

                # create a directory for each initialisation
                plot_init_dir = plot_dir + "/" + n_init + "/"
                import os

                if not os.path.exists(plot_init_dir):
                    from os import mkdir

                    mkdir(plot_init_dir)

                # save plot
                p.save(filename=plot_init_dir + step_name + ".png", limitsize=False)

    def plot_biol_spot_nUMI(self, fact_name="nUMI_factors"):
        r"""Plot the histogram of log10 of the sum across w_sf for each location

        Parameters
        ----------
        fact_name :
            parameter of the model to use plot (Default value = 'nUMI_factors')

        """

        plt.hist(np.log10(self.samples["post_sample_means"][fact_name].sum(1)), bins=50)
        plt.xlabel("Biological spot nUMI (log10)")
        plt.title("Biological spot nUMI")
        plt.tight_layout()

    def plot_spot_add(self):
        r"""Plot the histogram of log10 of additive location background."""

        plt.hist(np.log10(self.samples["post_sample_means"]["spot_add"][:, 0]), bins=50)
        plt.xlabel("UMI unexplained by biological factors")
        plt.title("Additive technical spot nUMI")
        plt.tight_layout()

    def plot_gene_E(self):
        r"""Plot the histogram of 1 / sqrt(overdispersion alpha)"""

        plt.hist((self.samples["post_sample_means"]["gene_E"][:, 0]), bins=50)
        plt.xlabel("E_g overdispersion parameter")
        plt.title("E_g overdispersion parameter")
        plt.tight_layout()

    def plot_gene_add(self):
        r"""Plot the histogram of additive gene background."""

        plt.hist((self.samples["post_sample_means"]["gene_add"][:, 0]), bins=50)
        plt.xlabel("S_g additive background noise parameter")
        plt.title("S_g additive background noise parameter")
        plt.tight_layout()

    def plot_gene_level(self):
        r"""Plot the histogram of log10 of M_g change in sensitivity between technologies."""

        plt.hist(np.log10(self.samples["post_sample_means"]["gene_level"][:, 0]), bins=50)
        plt.xlabel("M_g expression level scaling parameter")
        plt.title("M_g expression level scaling parameter")
        plt.tight_layout()

    def compute_expected(self):
        r"""Compute expected expression of each gene in each spot (Poisson mu). Useful for evaluating how well
        the model learned expression pattern of all genes in the data.
        """

        # compute the poisson rate
        self.mu = (
            np.dot(
                self.samples["post_sample_means"]["spot_factors"], self.samples["post_sample_means"]["gene_factors"].T
            )
            * self.samples["post_sample_means"]["gene_level"].T
            + self.samples["post_sample_means"]["gene_add"].T
            + self.samples["post_sample_means"]["spot_add"]
        )

    def compute_expected_clust(self, fact_ind):
        r"""Compute expected expression of each gene in each spot that comes from one reference cluster (Poisson mu)."""

        # compute the poisson rate
        self.mu = (
            np.dot(
                self.samples["post_sample_means"]["spot_factors"][:, fact_ind],
                self.samples["post_sample_means"]["gene_factors"].T[fact_ind, :],
            )
            * self.samples["post_sample_means"]["gene_level"].T
        )
