# -*- coding: utf-8 -*-
r"""This Location model decomposes the expression of genes across locations into a set
    of reference regulatory programmes, while accounting for correlation of programs
    across locations with similar cell composition.

    Unlike the standard LocationModel recommended for Visium data, this model includes a
    non-specific binding term that scales linearly with the total number of counts in the region
    of interest. In addition, it expects negative probe counts in each region of interest to estimate
    a prior distribution for this background."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from matplotlib.pyplot import figure

from cell2location.models.base.pymc3_loc_model import Pymc3LocModel


# defining the model itself
class LocationModelWTA(Pymc3LocModel):
    r"""Here we model the elements of :math:`D` as Poisson distributed,
    given an unobserved rate :math:`mu` and a gene-specific over-dispersion parameter :math:`\alpha_g`
    which describes variance in expression of individual genes that is not explained by the regulatory programs:

    .. math::
        D_{s,g} \sim \mathtt{Poisson}(\mu_{s,g})

    The spatial expression levels of genes :math:`\mu_{s,g}` in the rate space are modelled
    as the sum of five non-negative components:

    .. math::
        \mu_{s,g} = m_{g} \left (\sum_{f} {w_{s,f} \: g_{f,g}} \right) + l_s + s_{g}*totalCounts_s

    Here, :math:`w_{s,f}` denotes regression weight of each program :math:`f` at location :math:`s` ;
    :math:`g_{f,g}` denotes the regulatory programmes :math:`f` of each gene :math:`g` - input to the model;
    :math:`m_{g}` denotes a gene-specific scaling parameter which accounts for difference
    in the global expression estimates between technologies;
    :math:`l_{s}` and :math:`s_{g}` are additive components that capture additive background variation
    that is not explained by the bi-variate decomposition.

    The prior distribution on :math:`w_{s,f}` is chosen to reflect the absolute scale and account for correlation of programs
    across locations with similar cell composition. This is done by inferring a hierarchical prior representing
    the co-located cell type combinations.

    This prior is specified using 3 `cell_number_prior` input parameters:

    * **cells_per_spot** is derived from examining the paired histology image to get an idea about
      the average nuclei count per location.

    * **factors_per_spot** reflects the number of regulatory programmes / cell types you expect to find in each location.

    * **combs_per_spot** prior tells the model how much co-location signal to expect between the programmes / cell types.

    A number close to `factors_per_spot` tells that all cell types have independent locations,
    and a number close 1 tells that each cell type is co-located with `factors_per_spot` other cell types.
    Choosing a number halfway in-between is a sensible default: some cell types are co-located with others but some stand alone.

    The prior distribution on :math:`m_{g}` is informed by the expected change in sensitivity from single cell to spatial
    technology, and is specified in `gene_level_prior`.

    Note
    ----
        `gene_level_prior` and `cell_number_prior` determine the absolute scale of :math:`w_{s,f}` density across locations,
        but have a very limited effect on the absolute count of mRNA molecules attributed to each cell type.
        Comparing your prior on **cells_per_spot** to average nUMI in the reference and spatial data helps to choose
        the gene_level_prior and guide the model to learn :math:`w_{s,f}` close to the true cell count.

    Parameters
    ----------
    cell_state_mat :
        Pandas data frame with gene programmes - genes in rows, cell types / factors in columns
    X_data :
        Numpy array of gene probe counts (cols) in spatial locations (rows)
    Y_data :
        Numpy array of negative probe counts (cols) in spatial locations (rows)
    n_comb :
        The number of co-located cell type combinations (in the prior).
        The model is fairly robust to this choice when the prior has low effect on location weights W
        (`spot_fact_mean_var_ratio` parameter is low), but please use the default unless know what you are doing (Default: 50)
    n_iter :
        number of training iterations
    learning_rate, data_type, total_grad_norm_constraint, ...:
        See parent class BaseModel for details.
    gene_level_prior :
        prior on change in sensitivity between single cell and spatial technology (**mean**),
        how much individual genes deviate from that (**sd**),

        * **mean** a good choice of this prior for 10X Visium data and 10X Chromium reference is between 1/3 and 1 depending
          on how well each experiment worked. A good choice for SmartSeq 2 reference is around ~ 1/10.
        * **sd** a good choice of this prior is **mean** / 2.
          Avoid setting **sd** >= **mean** because it puts a lot of weight on 0.
    gene_level_var_prior :
        Certainty in the gene_level_prior (mean_var_ratio)
        - by default the variance in our prior of mean and sd is equal to the mean and sd
        decreasing this number means having higher uncertainty in the prior
    cell_number_prior :
        prior on cell density parameter:

        * **cells_per_spot** - what is the average number of cells you expect per location? This could also be the nuclei
          count from the paired histology image segmentation.
        * **factors_per_spot** - what is the number of cell types
          number of factors expressed per location?
        * **combs_per_spot** - what is the average number of factor combinations per location?
          a number halfway in-between `factors_per_spot` and 1 is a sensible default
          Low numbers mean more factors are co-located with other factors.
    cell_number_var_prior :
        Certainty in the cell_number_prior (cells_mean_var_ratio, factors_mean_var_ratio,
        combs_mean_var_ratio)
        - by default the variance in the value of this prior is equal to the value of this itself.
        decreasing this number means having higher uncertainty in the prior
    phi_hyp_prior :
        prior on NB alpha overdispersion parameter, the rate of exponential distribution over alpha.
        This is a containment prior so low values mean low deviation from the mean of NB distribution.

        * **mu** average prior
        * **sd** standard deviation in this prior
        When using the Visium data model is not sensitive to the choice of this prior so it is better to use the default.
    spot_fact_mean_var_ratio :
        the parameter that controls the strength of co-located cell combination prior on
        :math:`w_{s,f}` density across locations. It is expressed as mean / variance ratio with low values corresponding to
        a weakly informative prior. Use the default value of 0.5 unless you know what you are doing.

    Returns
    -------

    """

    def __init__(
        self,
        cell_state_mat: np.ndarray,
        X_data: np.ndarray,
        Y_data: np.ndarray,
        n_comb: int = 50,
        data_type: str = "float32",
        n_iter=20000,
        learning_rate=0.005,
        total_grad_norm_constraint=200,
        verbose=True,
        var_names=None,
        var_names_read=None,
        obs_names=None,
        fact_names=None,
        sample_id=None,
        gene_level_prior={"mean": 1 / 2, "sd": 1 / 4},
        gene_level_var_prior={"mean_var_ratio": 1},
        cell_number_prior={"cells_per_spot": 8, "factors_per_spot": 7, "combs_per_spot": 2.5},
        cell_number_var_prior={"cells_mean_var_ratio": 1, "factors_mean_var_ratio": 1, "combs_mean_var_ratio": 1},
        phi_hyp_prior={"mean": 3, "sd": 1},
        spot_fact_mean_var_ratio=0.5,
    ):

        ############# Initialise parameters ################
        super().__init__(
            cell_state_mat,
            X_data,
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

        self.Y_data = Y_data
        self.n_npro = Y_data.shape[1]
        self.y_data = theano.shared(Y_data.astype(self.data_type))
        self.n_rois = Y_data.shape[0]
        self.n_genes = X_data.shape[1]
        # Total number of gene counts in each region of interest, divided by 10^5:
        self.l_r = np.array([np.sum(X_data[i, :]) for i in range(self.n_rois)]).reshape(self.n_rois, 1) * 10 ** (-5)

        for k in gene_level_var_prior.keys():
            gene_level_prior[k] = gene_level_var_prior[k]

        self.gene_level_prior = gene_level_prior
        self.phi_hyp_prior = phi_hyp_prior
        self.n_comb = n_comb
        self.spot_fact_mean_var_ratio = spot_fact_mean_var_ratio

        cell_number_prior["factors_per_combs"] = (
            cell_number_prior["factors_per_spot"] / cell_number_prior["combs_per_spot"]
        )
        for k in cell_number_var_prior.keys():
            cell_number_prior[k] = cell_number_var_prior[k]
        self.cell_number_prior = cell_number_prior

        ############# Define the model ################
        self.model = pm.Model()

        with self.model:

            # ============================ Negative Probe Binding ===================== #
            # Negative probe counts scale linearly with the total number of counts in a region of interest.
            # The linear slope is drawn from a gamma distribution. Mean and variance are inferred from the data
            # and are the same for the non-specific binding term for gene probes further below.
            self.b_n_hyper = pm.Gamma("b_n_hyper", alpha=np.array((3, 1)), beta=np.array((1, 1)), shape=2)
            self.b_n = pm.Gamma("b_n", mu=self.b_n_hyper[0], sigma=self.b_n_hyper[1], shape=(1, self.n_npro))
            self.y_rn = self.b_n * self.l_r

            # ===================== Non-specific binding additive component ======================= #
            # Additive term for non-specific binding of gene probes are drawn from a gamma distribution with
            # the same mean and variance as for negative probes above.
            self.gene_add = pm.Gamma("gene_add", mu=self.b_n_hyper[0], sigma=self.b_n_hyper[1], shape=(1, self.n_genes))

            # =====================Gene expression level scaling======================= #
            # Explains difference in expression between genes and
            # how it differs in single cell and spatial technology
            # compute hyperparameters from mean and sd
            shape = gene_level_prior["mean"] ** 2 / gene_level_prior["sd"] ** 2
            rate = gene_level_prior["mean"] / gene_level_prior["sd"] ** 2
            shape_var = shape / gene_level_prior["mean_var_ratio"]
            rate_var = rate / gene_level_prior["mean_var_ratio"]
            self.gene_level_alpha_hyp = pm.Gamma(
                "gene_level_alpha_hyp", mu=shape, sigma=np.sqrt(shape_var), shape=(1, 1)
            )
            self.gene_level_beta_hyp = pm.Gamma("gene_level_beta_hyp", mu=rate, sigma=np.sqrt(rate_var), shape=(1, 1))

            self.gene_level = pm.Gamma(
                "gene_level", self.gene_level_alpha_hyp, self.gene_level_beta_hyp, shape=(self.n_genes, 1)
            )

            self.gene_factors = pm.Deterministic("gene_factors", self.cell_state)

            # =====================Spot factors======================= #
            # prior on spot factors reflects the number of cells, fraction of their cytoplasm captured,
            # times heterogeniety in the total number of mRNA between individual cells with each cell type
            self.cells_per_spot = pm.Gamma(
                "cells_per_spot",
                mu=cell_number_prior["cells_per_spot"],
                sigma=np.sqrt(cell_number_prior["cells_per_spot"] / cell_number_prior["cells_mean_var_ratio"]),
                shape=(self.n_rois, 1),
            )
            self.comb_per_spot = pm.Gamma(
                "combs_per_spot",
                mu=cell_number_prior["combs_per_spot"],
                sigma=np.sqrt(cell_number_prior["combs_per_spot"] / cell_number_prior["combs_mean_var_ratio"]),
                shape=(self.n_rois, 1),
            )

            shape = self.comb_per_spot / np.array(self.n_comb).reshape((1, 1))
            rate = tt.ones((1, 1)) / self.cells_per_spot * self.comb_per_spot
            self.combs_factors = pm.Gamma("combs_factors", alpha=shape, beta=rate, shape=(self.n_rois, self.n_comb))

            self.factors_per_combs = pm.Gamma(
                "factors_per_combs",
                mu=cell_number_prior["factors_per_combs"],
                sigma=np.sqrt(cell_number_prior["factors_per_combs"] / cell_number_prior["factors_mean_var_ratio"]),
                shape=(self.n_comb, 1),
            )
            c2f_shape = self.factors_per_combs / np.array(self.n_fact).reshape((1, 1))
            self.comb2fact = pm.Gamma(
                "comb2fact", alpha=c2f_shape, beta=self.factors_per_combs, shape=(self.n_comb, self.n_fact)
            )

            self.spot_factors = pm.Gamma(
                "spot_factors",
                mu=pm.math.dot(self.combs_factors, self.comb2fact),
                sigma=pm.math.sqrt(pm.math.dot(self.combs_factors, self.comb2fact) / self.spot_fact_mean_var_ratio),
                shape=(self.n_rois, self.n_fact),
            )

            # =====================Spot-specific additive component======================= #
            # molecule contribution that cannot be explained by cell state signatures
            # these counts are distributed between all genes not just expressed genes
            self.spot_add_hyp = pm.Gamma("spot_add_hyp", 1, 1, shape=2)
            self.spot_add = pm.Gamma("spot_add", self.spot_add_hyp[0], self.spot_add_hyp[1], shape=(self.n_rois, 1))

            # =====================Gene-specific overdispersion ======================= #
            self.phi_hyp = pm.Gamma("phi_hyp", mu=phi_hyp_prior["mean"], sigma=phi_hyp_prior["sd"], shape=(1, 1))
            self.gene_E = pm.Exponential("gene_E", self.phi_hyp, shape=(self.n_genes, 1))

            # =====================Expected expression ======================= #
            # Expected counts for negative probes and gene probes concatenated into one array. Note that non-specific binding
            # scales linearly with the total number of counts (l_r) in this model.
            self.mu_biol = tt.concatenate(
                [
                    self.y_rn,
                    pm.math.dot(self.spot_factors, self.gene_factors.T) * self.gene_level.T
                    + self.gene_add * self.l_r
                    + self.spot_add,
                ],
                axis=1,
            )

            # =====================DATA likelihood ======================= #
            # Likelihood (sampling distribution) of observations & add overdispersion via NegativeBinomial / Poisson
            self.data_target = pm.NegativeBinomial(
                "data_target",
                mu=self.mu_biol,
                alpha=tt.concatenate(
                    [np.repeat(10 ** 10, self.n_npro).reshape(1, self.n_npro), 1 / (self.gene_E.T * self.gene_E.T)],
                    axis=1,
                ),
                observed=tt.concatenate([self.y_data, self.x_data], axis=1),
            )

            # =====================Compute nUMI from each factor in spots  ======================= #
            self.nUMI_factors = pm.Deterministic(
                "nUMI_factors", (self.spot_factors * (self.gene_factors * self.gene_level).sum(0))
            )

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
            + self.samples["post_sample_means"]["gene_add"] * self.l_r
            + self.samples["post_sample_means"]["spot_add"]
        )

    def plot_Locations_1D_scatterPlot(
        self, x, order=None, polynomial_order=6, figure_size=(30, 30), saveFig=None, density=True, xlabel="x-coordinate"
    ):

        # Set figure parameters:
        SMALL_SIZE = 20
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 20

        plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

        figure(num=None, figsize=figure_size, dpi=80, facecolor="w", edgecolor="k")

        cellColours = np.array(
            (
                "blue",
                "red",
                "purple",
                "yellow",
                "green",
                "blue",
                "purple",
                "yellow",
                "red",
                "green",
                "blue",
                "purple",
                "yellow",
                "red",
                "green",
                "blue",
                "purple",
                "yellow",
                "red",
            )
        )
        for i in range(len(self.fact_names)):
            plt.subplot(np.ceil(np.sqrt(len(self.fact_names))), np.ceil(np.sqrt(len(self.fact_names))), i + 1)
            results = np.array(self.spot_factors_df)[order, :]
            if density:
                results = (
                    results.T
                    / [
                        sum(
                            results[
                                i,
                            ]
                        )
                        for i in range(len(results[:, 1]))
                    ]
                ).T
            y = results[:, self.fact_names == self.fact_names[i]][:, 0]
            plt.plot(
                np.unique(x[order]),
                np.poly1d(np.polyfit(x[order], y, polynomial_order))(np.unique(x[order])),
                c=cellColours[0],
            )
            plt.scatter(x[order], y, c=cellColours[0], s=100)
            plt.xlabel(xlabel)
            if density:
                plt.ylabel("Cell Type Density")
            else:
                plt.ylabel("Cell Type Number")
            plt.title(self.fact_names[i])
        plt.tight_layout()
        if saveFig:
            plt.savefig(saveFig)
        plt.show()

    def plot_Locations_1D_scatterPlot_multipleCategories(
        self,
        x,
        order=None,
        polynomial_order=6,
        figure_size=(30, 30),
        saveFig=None,
        density=True,
        xlabel="x-coordinate",
        categories=[None],
    ):

        # Set figure parameters:
        SMALL_SIZE = 20
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 20

        plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

        figure(num=None, figsize=figure_size, dpi=80, facecolor="w", edgecolor="k")

        cellColours = np.array(
            (
                "blue",
                "red",
                "purple",
                "yellow",
                "green",
                "blue",
                "purple",
                "yellow",
                "red",
                "green",
                "blue",
                "purple",
                "yellow",
                "red",
                "green",
                "blue",
                "purple",
                "yellow",
                "red",
            )
        )
        for i in range(len(self.fact_names)):
            plt.subplot(np.ceil(np.sqrt(len(self.fact_names))), np.ceil(np.sqrt(len(self.fact_names))), i + 1)
            for j in range(len(categories)):
                results_j = np.array(self.spot_factors_df)[order[j], :]
                if density:
                    results_j = (
                        results_j.T
                        / [
                            sum(
                                results_j[
                                    i,
                                ]
                            )
                            for i in range(len(results_j[:, 1]))
                        ]
                    ).T
                y = results_j[:, self.fact_names == self.fact_names[i]][:, 0]
                x_j = x[order[j]]
                plt.plot(
                    np.unique(x_j),
                    np.poly1d(np.polyfit(x_j, y, polynomial_order))(np.unique(x_j)),
                    label=categories[j],
                    c=cellColours[j],
                )
                plt.scatter(x_j, y, c=cellColours[j], s=100)
            plt.xlabel(xlabel)
            if density:
                plt.ylabel("Cell Type Density")
            else:
                plt.ylabel("Cell Type Number")
            plt.legend()
            plt.title(self.fact_names[i])
        plt.tight_layout()
        if saveFig:
            plt.savefig(saveFig)
        plt.show()

    def plot_prior_sample(self):

        fig, ax = plt.subplots(2, 3, figsize=(10, 15))

        data_node = "X_data"
        data_target_name = "data_target"

        if type(data_node) is str:
            data_node = getattr(self, data_node)

        if type(data_target_name) is str:
            data_target_name = self.prior_trace[data_target_name][:, :, self.n_npro :]

        # If there are multiple prior samples, expand the data array
        if len(data_target_name.shape) > 2:
            data_node = np.array([data_node for _ in range(data_target_name.shape[0])])

        data_node = np.log10(data_node + 1)
        data_target_name = np.log10(data_target_name + 1)

        ax[0, 0].hist2d(data_node.flatten(), data_target_name.flatten(), bins=50, norm=matplotlib.colors.LogNorm())
        ax[0, 0].set_xlabel("X_data observed, log10(nUMI)")
        ax[0, 0].set_ylabel("X_data prior, log10(nUMI)")
        ax[0, 0].set_title("X_data prior vs X_data observed")

        ax[0, 1].hist(data_node.flatten(), bins=50)
        ax[0, 1].set_xlabel("X_data observed, log10(nUMI)")
        ax[0, 1].set_ylabel("Occurences")
        ax[0, 1].set_title("X_data observed")

        ax[0, 2].hist(data_target_name.flatten(), bins=50)
        ax[0, 2].set_xlabel("X_data prior, log10(nUMI)")
        ax[0, 2].set_ylabel("Occurences")
        ax[0, 2].set_title("X_data prior")

        data_node = "Y_data"
        data_target_name = "data_target"

        if type(data_node) is str:
            data_node = getattr(self, data_node)

        if type(data_target_name) is str:
            data_target_name = self.prior_trace[data_target_name][:, :, : self.n_npro]

        # If there are multiple prior samples, expand the data array
        if len(data_target_name.shape) > 2:
            data_node = np.array([data_node for _ in range(data_target_name.shape[0])])

        data_node = np.log10(data_node + 1)
        data_target_name = np.log10(data_target_name + 1)

        ax[1, 0].hist2d(data_node.flatten(), data_target_name.flatten(), bins=50, norm=matplotlib.colors.LogNorm())
        ax[1, 0].set_xlabel("Y_data observed, log10(nUMI)")
        ax[1, 0].set_ylabel("Y_data prior, log10(nUMI)")
        ax[1, 0].set_title("Y_data prior vs Y_data observed")

        ax[1, 1].hist(data_node.flatten(), bins=50)
        ax[1, 1].set_xlabel("Y_data observed, log10(nUMI)")
        ax[1, 1].set_ylabel("Occurences")
        ax[1, 1].set_title("Y_data observed")

        ax[1, 2].hist(data_target_name.flatten(), bins=50)
        ax[1, 2].set_xlabel("Y_data prior, log10(nUMI)")
        ax[1, 2].set_ylabel("Occurences")
        ax[1, 2].set_title("Y_data prior")

        plt.tight_layout()
