# -*- coding: utf-8 -*-
"""Negative binomial regression model that accounts for multiplicative sample scaling and additive sample background
(MLE/MAP in pytorch).

To address the usage of challenging reference data composed of multiple batches :math:`e=\{1..E\}`
the expression of each gene `g` in each single cell reference cluster `f` (possibly other meaningful covariates)
is inferred using a regularised Negative Binomial regression.

The model accounts for the sequencing coverage :math:`h_e` and background expression :math:`b_{eg}` in each sample `e`.
It models unexplained variance (overdispersion :math:`\\alpha_g`) and count nature of the data using
the Negative Binomial distribution:

.. math::
    J_{cg} \sim \mathtt{NB}(\mu_{cg}, 1 / \\alpha_g^2)

.. math::
    \mu_{cg} = (g_{fg} + b_{eg}) \: {h_e}

All model parameters are constrained to be positive to simplify interpretation.
Weak L2 regularisation of :math:`g_{fg}` / :math:`b_{eg}` / :math:`\\alpha_g` and penalty for large deviations of
:math:`h_e` from 1 is used. :math:`g_{fg}` is initialised at analytical average for each `f`,
:math:`b_{eg}` is initialised at average expression of each gene `g` in each sample `e` divided by a factor of 10.
The informative initialisation leads to fast convergence.

Training can be performed using mini batches of cells (30sec-5min on GPU) or on full data.
"""

# +
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cell2location.models.base.regression_torch_model import RegressionTorchModel


# defining the model itself
class RegressionGeneBackgroundCoverageModule(nn.Module):
    """Module class that defines the model graph and parameters.
    A module class should have the following methods:

      * __init__ which defines parameters (real scale)
      * forward which uses input data and parameters to compute parameters of data-generating distribution
        (e.g. NB mean and alpha)
      * initialize_parameters which initialises parameters (real scale)
      * export_parameters which transforms parameters correct domain (e.g. positive)

    Parameters
    ----------
    n_cells :
        number of cells / spots / observations
    n_genes :
        number of genes / variables
    n_fact :
        number of factors / covariates / reference signatures (extracted from `cell_state_mat.shape[1]`)
    clust_average_mat :
        initial value of factors / covariates (if None, initialised as normal_(mean=0, std=1/2)
    which_sample :
        boolean array, which covariates denote sample / experiment
    data_type :
        np.dtype to use (Default 'float32')
    eps :
        numerical stability constant

    """

    def __init__(self, n_cells, n_genes, n_fact, clust_average_mat, which_sample, data_type="float32", eps=1e-8):

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
        """Computes NB distribution parameters using input covariates for each cell:
        expected value of expression (mu_biol, cells (minibatch * genes) and overdispersion (gene_E).

        Parameters
        ----------
        cell2sample_covar :
            np.narray that gives sample membership (first X columns)
            and covariate values (all other columns) for each cell / observation (rows) - could be done in in mini batches.

        Returns
        -------
        list
            a list with [mu_biol, gene_E]

        """

        # sample-specific scaling of expression levels
        self.cell2sample_scaling = torch.mm(cell2sample_covar[:, self.which_sample], torch.exp(self.sample_scaling_log))
        self.mu_biol = torch.mm(cell2sample_covar, torch.exp(self.gene_factors_log)) * self.cell2sample_scaling

        self.gene_E = torch.tensor(1) / (torch.exp(self.gene_E_log) * torch.exp(self.gene_E_log))

        return [self.mu_biol, self.gene_E]

    def initialize_parameters(self):
        """Initialise parameters, using `clust_average_mat` for covariate and sample effects,
        random initialisation in a sensible range (see code) for all other parameters.

        Parameters
        ----------
        """

        if self.clust_average_mat is not None:
            self.gene_factors_log.data = nn.Parameter(
                torch.tensor(np.log(self.clust_average_mat.astype(self.data_type) + self.eps))
            )
        else:
            self.gene_factors_log.data.normal_(mean=0, std=1 / 2)

        self.sample_scaling_log.data.normal_(mean=0, std=1 / 20)
        self.gene_E_log.data.normal_(mean=-1 / 2, std=1 / 2)

    def export_parameters(self):
        """Compute parameters from their real number representation.

        Returns
        -------
        dict
            a dictionary with parameters {'gene_factors', 'sample_scaling', 'gene_E'}
        """

        export = {
            "gene_factors": torch.exp(self.gene_factors_log),
            "sample_scaling": torch.exp(self.sample_scaling_log),
            "gene_E": torch.exp(self.gene_E_log),
        }

        return export


class RegressionGeneBackgroundCoverageTorch(RegressionTorchModel):
    """Negative binomial regression model that accounts for multiplicative sample scaling
    and additive sample background (MLE/MAP in pytorch).

    Parameters
    ----------
    sample_id :
        str with column name in cell2covar that denotes sample
    cell2covar :
        pd.DataFrame with covariates in columns and cells in rows, rows should be named.
    X_data :
        Numpy array of gene expression (cols) in cells (rows)
    n_iter :
        number of iterations, when using minibatch, the number of epochs (passes through all data),
        supersedes self.n_iter
    """

    # RegressionNBV2Torch
    def __init__(
        self,
        sample_id,
        cell2covar: pd.DataFrame,
        X_data: np.ndarray,
        data_type="float32",
        n_iter=50000,
        learning_rate=0.005,
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
        super().__init__(
            sample_id,
            cell2covar,
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
            minibatch_size,
            minibatch_seed,
            prior_eps,
            nb_param_conversion_eps,
            use_cuda,
            use_average_as_initial_value,
            stratify_cv,
        )

        ############# Define the model ################

        self.model = RegressionGeneBackgroundCoverageModule(
            self.n_obs,
            self.n_var,
            self.n_fact,
            self.clust_average_mat,
            which_sample=self.which_sample,
            data_type=self.data_type,
        )

    # =====================DATA likelihood loss function======================= #
    # define cost function
    def loss(self, param, data, l2_weight=None):
        """Method that returns NB loss + L2 penalty on parameters (a single number)

        Parameters
        ----------
        param :
            list with NB distribution parameters[mu, theta/alpha], see self.nb_log_prob for details
        data :
            data (cells * genes), see self.nb_log_prob for details
        l2_weight :
            strength of L2 regularisation for each parameter that needs distinct regularisation
            (computed on exported parameters), if None - no regularisation :

            * **l2_weight** - for all parameters but mainly for co-variate effects, this should be very weak to not
              over-penalise good solution (Default: 0.001)
            * **sample_scaling_weight** - strong penalty for deviation from 1 (Default: 0.5)
            * **gene_overdisp_weight** - gene overdispersion penalty to provide containment prior
              (Default: 0.001 but will be changed)

        Returns
        -------

        """

        # initialise extra penalty
        l2_reg = 0

        if l2_weight is not None:

            # set default parameters
            d_l2_weight = {"l2_weight": 0.001, "sample_scaling_weight": 0.5, "gene_overdisp_weight": 0.001}
            # replace defaults with parameters supplied
            if l2_weight:  # if True use all defaults
                l2_weight = d_l2_weight
            else:
                for k in l2_weight.keys():
                    d_l2_weight[k] = l2_weight[k]

            param_dict = self.model.export_parameters()
            sample_scaling = param_dict["sample_scaling"]
            gene_e = param_dict["gene_E"]

            param_val = [param_dict[i] for i in param_dict.keys() if i not in ["sample_scaling", "gene_E"]]

            # regularise sample_scaling
            l2_reg = l2_reg + l2_weight["sample_scaling_weight"] * (sample_scaling - 1).pow(2).sum()
            # regularise overdispersion
            l2_reg = l2_reg + l2_weight["gene_overdisp_weight"] * gene_e.pow(2).sum()

            # regularise other parameters
            for i in param_val:
                l2_reg = l2_reg + l2_weight["l2_weight"] * i.pow(2).sum()

        return -self.nb_log_prob(param, data).sum() + l2_reg

    # =====================Other functions======================= #

    def evaluate_stability(self, n_samples=1000, align=True, transpose=True):
        r"""Evaluate stability of point estimates between training initialisations
        (correlates the values of factors between training initialisations)
        See TorchModel.b_evaluate_stability for argument details (node_name='gene_factors' here).
        """
        self.b_evaluate_stability("gene_factors", n_samples=n_samples, align=align, transpose=transpose)

    def compute_expected(self):
        """Compute expected expression of each gene in each cell (Poisson mu)."""

        # compute the poisson rate
        self.mu = np.dot(self.cell2sample_covar_mat, self.samples["post_sample_means"]["gene_factors"]) * np.dot(
            self.cell2sample_mat, self.samples["post_sample_means"]["sample_scaling"]
        )

    def compute_expected_fact(self, fact_ind, sample_scaling=True):
        """Compute expected expression of each gene in each cell
        that comes from a subset of factors (Poisson mu). E.g. expressed factors in self.fact_filt (by default)

        Parameters
        ----------
        sample_scaling :
            if False do not rescale levels to specific sample (Default value = True)
        fact_ind :
            boolean or integer array selecting covariates for reconstruction (to include all use .compute_expected())

        """

        # compute the poisson rate
        self.mu = np.dot(
            self.cell2sample_covar_mat[:, fact_ind], self.samples["post_sample_means"]["gene_factors"][fact_ind, :]
        )
        if sample_scaling:
            self.mu = self.mu * np.dot(self.cell2sample_mat, self.samples["post_sample_means"]["sample_scaling"])

    def normalise(self, X_data, remove_additive=True, remove_sample_scaling=True):
        """Normalise expression data by inferred sample scaling parameters
        and remove additive sample background.

        Parameters
        ----------
        X_data:
             Data to normalise
        remove_additive :
             (Default value = True)
        remove_sample_scaling :
             (Default value = True)

        """

        corrected = X_data.copy()
        if remove_sample_scaling:
            corrected = corrected / np.dot(self.cell2sample_mat, self.samples["post_sample_means"]["sample_scaling"])

        if remove_additive:
            # remove additive sample effects
            corrected = corrected - np.dot(self.cell2sample_mat, self.sample_effects.values.T)

        corrected = corrected - corrected.min()

        return corrected
