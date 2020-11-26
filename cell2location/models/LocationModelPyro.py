r"""Location model decomposes the expression of genes across locations into a set of reference regulatory programmes,
    it is identical to LocationModelLinearDependentWPyro but does not account for linear dependencies in abundance of programs
    across locations with similar cell composition."""

import numpy as np

import pyro
import pyro.distributions as dist
from pyro import poutine

from cell2location.models.pyro_loc_model import PyroLocModel
from cell2location.distributions.NegativeBinomial import NegativeBinomial

import torch


def rand_tensor(shape, mean, sigma):
    r""" Helper for initializing variational parameters
    """
    return mean * torch.ones(shape) + sigma * torch.randn(shape)


########-------- defining the model itself - pyro -------- ########
class LocationModelPyro(PyroLocModel):
    r"""Provided here as a 'base' model for completeness.

    Parameters
    ----------
    cell_state_mat :
        Pandas data frame with gene programmes - genes in rows, cell types / factors in columns
    X_data :
        Numpy array of gene expression (cols) in spatial locations (rows)
    n_iter :
        number of training iterations
    learning_rate, data_type, total_grad_norm_constraint, ...:
        See parent class BaseModel for details.
    gene_level_prior :
        see the description for LocationModelLinearDependentWPyro
    gene_level_var_prior :
        see the description for LocationModelLinearDependentWPyro
    cell_number_prior :
        see the description for LocationModelLinearDependentWPyro, this model does not have **combs_per_spot**
        parameter.
    cell_number_var_prior :
        see the description for LocationModelLinearDependentWPyro, this model does not have
        **combs_mean_var_ratio** parameter.
    phi_hyp_prior :
        see the description for LocationModelLinearDependentWPyro

    Returns
    -------

    """

    def __init__(
            self,
            cell_state_mat: np.ndarray,
            X_data: np.ndarray,
            data_type: str = 'float32',
            n_iter=15000,
            learning_rate=0.005,
            total_grad_norm_constraint=200,
            use_cuda=True,
            verbose=True,
            var_names=None, var_names_read=None,
            obs_names=None, fact_names=None, sample_id=None,
            gene_level_prior={'mean': 1 / 2, 'sd': 1 / 4},
            gene_level_var_prior={'mean_var_ratio': 1},
            cell_number_prior={'cells_per_spot': 8,
                               'factors_per_spot': 7},
            cell_number_var_prior={'cells_mean_var_ratio': 1,
                                   'factors_mean_var_ratio': 1},
            phi_hyp_prior={'mean': 3, 'sd': 1},
            initialise_at_prior=False,
            minibatch_size=None,
            minibatch_seed=42
    ):

        ############# Initialise parameters ################
        super().__init__(cell_state_mat=cell_state_mat, X_data=X_data,
                         data_type=data_type, n_iter=n_iter,
                         learning_rate=learning_rate, total_grad_norm_constraint=total_grad_norm_constraint,
                         use_cuda=use_cuda, verbose=verbose, var_names=var_names, var_names_read=var_names_read,
                         obs_names=obs_names, fact_names=fact_names, sample_id=sample_id,
                         minibatch_size=minibatch_size, minibatch_seed=minibatch_seed)

        self.gene_level_prior = gene_level_prior
        self.cell_number_prior = cell_number_prior
        self.phi_hyp_prior = phi_hyp_prior
        self.initialise_at_prior = initialise_at_prior

        for k in cell_number_var_prior.keys():
            cell_number_prior[k] = cell_number_var_prior[k]
        self.cell_number_prior = cell_number_prior

        for k in gene_level_var_prior.keys():
            gene_level_prior[k] = gene_level_var_prior[k]
        self.gene_level_prior = gene_level_prior

        ## define parameters used to initialize variational parameters##
        self.gl_shape = self.gene_level_prior['mean'] ** 2 / self.gene_level_prior['sd'] ** 2
        self.gl_rate = self.gene_level_prior['mean'] / self.gene_level_prior['sd'] ** 2
        self.gl_shape_var = self.gl_shape / self.gene_level_prior['mean_var_ratio']
        self.gl_rate_var = self.gl_rate / self.gene_level_prior['mean_var_ratio']
        self.gl_mean = self.gene_level_prior['mean']
        self.gl_sd = self.gene_level_prior['sd']

    ############# Define the model ################
    def model(self, x_data, idx=None):


        # =====================Gene expression level scaling======================= #
        # Explains difference in expression between genes and
        # how it differs in single cell and spatial technology
        # compute hyperparameters from mean and sd
        gl_alpha_shape = self.gl_shape ** 2 / self.gl_shape_var
        gl_alpha_rate = self.gl_shape / self.gl_shape_var
        gl_beta_shape = self.gl_rate ** 2 / self.gl_rate_var
        gl_beta_rate = self.gl_rate / self.gl_rate_var

        self.gene_level_alpha_hyp = pyro.sample('gene_level_alpha_hyp',
                                                dist.Gamma(torch.ones([1, 1]) * torch.tensor(gl_alpha_shape),
                                                           torch.ones([1, 1]) * torch.tensor(gl_alpha_rate)))
        self.gene_level_beta_hyp = pyro.sample('gene_level_beta_hyp',
                                               dist.Gamma(torch.ones([1, 1]) * torch.tensor(gl_beta_shape),
                                                          torch.ones([1, 1]) * torch.tensor(gl_beta_rate)))

        self.gene_level = pyro.sample('gene_level',
                                      dist.Gamma(torch.ones([self.n_var, 1]) * self.gene_level_alpha_hyp,
                                                 torch.ones([self.n_var, 1]) * self.gene_level_beta_hyp))

        # scale cell state factors by gene_level
        self.gene_factors = pyro.deterministic('gene_factors', self.cell_state)

        # =====================Spot factors======================= #
        # prior on spot factors reflects the number of cells, fraction of their cytoplasm captured,
        # times heterogeniety in the total number of mRNA between individual cells with each cell type
        cps_shape = self.cell_number_prior['cells_per_spot'] ** 2 \
                    / (self.cell_number_prior['cells_per_spot'] / self.cell_number_prior['cells_mean_var_ratio'])
        cps_rate = self.cell_number_prior['cells_per_spot'] \
                   / (self.cell_number_prior['cells_per_spot'] / self.cell_number_prior['cells_mean_var_ratio'])
        self.cells_per_spot = pyro.sample('cells_per_spot',
                                          dist.Gamma(torch.ones([self.n_obs, 1]) * torch.tensor(cps_shape),
                                                     torch.ones([self.n_obs, 1]) * torch.tensor(cps_rate)))

        fps_shape = self.cell_number_prior['factors_per_spot'] ** 2 \
                    / (self.cell_number_prior['factors_per_spot'] / self.cell_number_prior['factors_mean_var_ratio'])
        fps_rate = self.cell_number_prior['factors_per_spot'] \
                   / (self.cell_number_prior['factors_per_spot'] / self.cell_number_prior['factors_mean_var_ratio'])
        self.factors_per_spot = pyro.sample('factors_per_spot',
                                            dist.Gamma(torch.ones([self.n_obs, 1]) * torch.tensor(fps_shape),
                                                       torch.ones([self.n_obs, 1]) * torch.tensor(fps_rate)))

        shape = self.factors_per_spot / torch.tensor(np.array(self.n_fact).reshape((1, 1)))
        rate = torch.ones([1, 1]) / self.cells_per_spot * self.factors_per_spot
        self.spot_factors = pyro.sample('spot_factors',
                                        dist.Gamma(torch.matmul(shape, torch.ones([1, self.n_fact])),
                                                   torch.matmul(rate, torch.ones([1, self.n_fact]))))

        # =====================Spot-specific additive component======================= #
        # molecule contribution that cannot be explained by cell state signatures
        # these counts are distributed between all genes not just expressed genes
        self.spot_add_hyp = pyro.sample('spot_add_hyp',
                                        dist.Gamma(torch.ones([2, 1]) * torch.tensor(1.),
                                                   torch.ones([2, 1]) * torch.tensor(0.1)))
        self.spot_add = pyro.sample('spot_add',
                                    dist.Gamma(torch.ones([self.n_obs, 1]) * self.spot_add_hyp[0, 0],
                                               torch.ones([self.n_obs, 1]) * self.spot_add_hyp[1, 0]))

        # =====================Gene-specific additive component ======================= #
        # per gene molecule contribution that cannot be explained by cell state signatures
        # these counts are distributed equally between all spots (e.g. background, free-floating RNA)
        self.gene_add_hyp = pyro.sample('gene_add_hyp',
                                        dist.Gamma(torch.ones([2, 1]) * torch.tensor(1.),
                                                   torch.ones([2, 1]) * torch.tensor(1.)))
        self.gene_add = pyro.sample('gene_add',
                                    dist.Gamma(torch.ones([self.n_var, 1]) * self.gene_add_hyp[0, 0],
                                               torch.ones([self.n_var, 1]) * self.gene_add_hyp[1, 0]))

        # =====================Gene-specific overdispersion ======================= #
        self.phi_hyp = pyro.sample('phi_hyp',
                                   dist.Gamma(torch.ones([1, 1]) * torch.tensor(self.phi_hyp_prior['mean']),
                                              torch.ones([1, 1]) * torch.tensor(self.phi_hyp_prior['sd'])))
        self.gene_E = pyro.sample('gene_E', dist.Exponential(torch.ones([self.n_var, 1]) * self.phi_hyp[0, 0]))

        # =====================Expected expression ======================= #
        # expected expression
        self.mu_biol = torch.matmul(self.spot_factors[idx], self.gene_factors.T) * self.gene_level.T \
                       + self.gene_add.T + self.spot_add[idx]

        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        self.data_target = pyro.sample('data_target',
                                       NegativeBinomial(mu=self.mu_biol,
                                                        theta=torch.ones([1, 1]) / (self.gene_E.T * self.gene_E.T)),
                                       obs=x_data)

        # =====================Compute nUMI from each factor in spots  ======================= #
        nUMI = (self.spot_factors * (self.gene_factors * self.gene_level).sum(0))
        self.nUMI_factors = pyro.deterministic('nUMI_factors', nUMI)

    def compute_expected(self):
        r"""Compute expected expression of each gene in each spot (Poisson mu). Useful for evaluating how well
        the model learned expression pattern of all genes in the data.

        """

        # compute the poisson rate
        self.mu = (np.dot(self.samples['post_sample_means']['spot_factors'],
                          self.samples['post_sample_means']['gene_factors'].T)
                   * self.samples['post_sample_means']['gene_level'].T
                   + self.samples['post_sample_means']['gene_add'].T
                   + self.samples['post_sample_means']['spot_add'])

    def step_train(self, name, x_data, extra_data):
        idx = extra_data.get('idx')
        if idx is None:
            idx = torch.LongTensor(np.arange(x_data.shape[0]))
        return self.svi[name].step(x_data, idx)

    def step_eval_loss(self, name, x_data, extra_data):
        idx = extra_data.get('idx')
        if idx is None:
            idx = torch.LongTensor(np.arange(x_data.shape[0]))
        return self.svi[name].evaluate_loss(x_data, idx)

    def step_predictive(self, predictive, x_data, extra_data):
        idx = extra_data.get('idx')
        if idx is None:
            idx = torch.LongTensor(np.arange(x_data.shape[0]))
        return predictive(x_data, idx)

    def step_trace(self, name, x_data, extra_data):
        idx = extra_data.get('idx')
        if idx is None:
            idx = torch.LongTensor(np.arange(x_data.shape[0]))
        guide_tr = poutine.trace(self.guide_i[name]).get_trace(x_data, idx)
        model_tr = poutine.trace(poutine.replay(self.model,
                                                trace=guide_tr)).get_trace(x_data, idx)
        return guide_tr, model_tr
