########----------------########
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine

from cell2location.models.pyro_model import PyroModel


def Gamma(mu=None, sigma=None, alpha=None, beta=None, shape=None):
    if alpha is not None and beta is not None:
        pass
    elif mu is not None and sigma is not None:
        alpha = mu ** 2 / sigma ** 2
        beta = mu / sigma ** 2
    else:
        raise ValueError('Define (mu and var) or (alpha and beta).')
    if shape is None:
        alpha = torch.tensor(alpha)
        beta = torch.tensor(beta)
    else:
        alpha = torch.ones(shape) * alpha
        beta = torch.ones(shape) * beta
    return dist.Gamma(alpha, beta)


def rand_tensor(shape, mean, sigma):
    r""" Helper for initializing variational parameters
    """
    return mean * torch.ones(shape) + sigma * torch.randn(shape)


def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""NB parameterizations conversion  - Copied over from scVI
        :param mu: mean of the NB distribution.
        :param theta: inverse overdispersion.
        :param eps: constant used for numerical log stability.
        :return: the number of failures until the experiment is stopped
            and the success probability.
    """
    assert (mu is None) == (
            theta is None
    ), "If using the mu/theta NB parameterization, both parameters must be specified"
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits


def _convert_counts_logits_to_mean_disp(total_count, logits):
    """NB parameterizations conversion  - Copied over from scVI
        :param total_count: Number of failures until the experiment is stopped.
        :param logits: success logits.
        :return: the mean and inverse overdispersion of the NB distribution.
    """
    theta = total_count
    mu = logits.exp() * theta
    return mu, theta


########-------- defining the model itself - pyro -------- ########
class LocationModelLinearDependentWPyro(PyroModel):
    r"""LocationModelPyro2NB4V7_V4_V4 Cell location model with E_g overdispersion &
    NB likelihood defined by total_count and logits
         - similar to LocationModelNB4V7_V4_V4
         pymc3 NB parametrisation of pytorch but overdisp priors as described here https://statmodeling.stat.columbia.edu/2018/04/03/justify-my-love/
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
            n_comb: int = 50,
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
                               'factors_per_spot': 7,
                               'combs_per_spot': 2.5},
            cell_number_var_prior={'cells_mean_var_ratio': 1,
                                   'factors_mean_var_ratio': 1,
                                   'combs_mean_var_ratio': 1},
            phi_hyp_prior={'mean': 3, 'sd': 1},
            spot_fact_mean_var_ratio=5,
            minibatch_size=None,
            minibatch_seed=42
    ):

        ############# Initialise parameters ################
        super().__init__(X_data, cell_state_mat.shape[1],
                         data_type, n_iter,
                         learning_rate, total_grad_norm_constraint,
                         use_cuda, verbose, var_names, var_names_read,
                         obs_names, fact_names, sample_id, minibatch_size,
                         minibatch_seed)

        self.cell_state_mat = cell_state_mat
        # Pass data to pyro / pytorch
        self.cell_state = torch.tensor(cell_state_mat.astype(self.data_type))  # .double()
        if self.use_cuda:
            # move tensors and modules to CUDA
            self.cell_state = self.cell_state.cuda()

        for k in gene_level_var_prior.keys():
            gene_level_prior[k] = gene_level_var_prior[k]

        self.gene_level_prior = gene_level_prior
        self.phi_hyp_prior = phi_hyp_prior
        self.n_comb = n_comb
        self.spot_fact_mean_var_ratio = spot_fact_mean_var_ratio

        cell_number_prior['factors_per_combs'] = (cell_number_prior['factors_per_spot'] /
                                                  cell_number_prior['combs_per_spot'])
        for k in cell_number_var_prior.keys():
            cell_number_prior[k] = cell_number_var_prior[k]
        self.cell_number_prior = cell_number_prior

    ############# Define the model ################
    def model(self, x_data, idx=None):

        # =====================Gene expression level scaling======================= #
        # Explains difference in expression between genes and
        # how it differs in single cell and spatial technology
        # compute hyperparameters from mean and sd

        shape = self.gene_level_prior['mean'] ** 2 / self.gene_level_prior['sd'] ** 2
        rate = self.gene_level_prior['mean'] / self.gene_level_prior['sd'] ** 2
        shape_var = shape / self.gene_level_prior['mean_var_ratio']
        rate_var = rate / self.gene_level_prior['mean_var_ratio']

        n_g_prior = np.array(self.gene_level_prior['mean']).shape
        if len(n_g_prior) == 0:
            n_g_prior = 1
        else:
            n_g_prior = self.n_genes

        self.gene_level_alpha_hyp = pyro.sample('gene_level_alpha_hyp',
                                                Gamma(mu=shape,
                                                      sigma=np.sqrt(shape_var),
                                                      shape=(n_g_prior, 1)))

        self.gene_level_beta_hyp = pyro.sample('gene_level_beta_hyp',
                                               Gamma(mu=rate,
                                                     sigma=np.sqrt(rate_var),
                                                     shape=(n_g_prior, 1)))

        self.gene_level = pyro.sample('gene_level',
                                      Gamma(alpha=self.gene_level_alpha_hyp,
                                            beta=self.gene_level_beta_hyp,
                                            shape=(self.n_genes, 1)))

        # scale cell state factors by gene_level
        self.gene_factors = pyro.deterministic('gene_factors', self.cell_state)

        # =====================Spot factors======================= #
        # prior on spot factors reflects the number of cells, fraction of their cytoplasm captured,
        # times heterogeneity in the total number of mRNA between individual cells with each cell type
        self.cells_per_spot = pyro.sample('cells_per_spot',
                                          Gamma(mu=self.cell_number_prior['cells_per_spot'],
                                                sigma=np.sqrt(self.cell_number_prior['cells_per_spot'] \
                                                              / self.cell_number_prior['cells_mean_var_ratio']),
                                                shape=(self.n_cells, 1)))

        self.comb_per_spot = pyro.sample('combs_per_spot',
                                         Gamma(mu=self.cell_number_prior['combs_per_spot'],
                                               sigma=np.sqrt(self.cell_number_prior['combs_per_spot'] \
                                                             / self.cell_number_prior['combs_mean_var_ratio']),
                                               shape=(self.n_cells, 1)))

        shape = self.comb_per_spot / self.n_comb
        rate = torch.ones([1, 1]) / self.cells_per_spot * self.comb_per_spot

        self.combs_factors = pyro.sample('combs_factors',
                                         Gamma(alpha=shape,
                                               beta=rate,
                                               shape=(self.n_cells, self.n_comb)))

        self.factors_per_combs = pyro.sample('factors_per_combs',
                                             Gamma(mu=self.cell_number_prior['factors_per_combs'],
                                                   sigma=np.sqrt(self.cell_number_prior['factors_per_combs'] \
                                                                 / self.cell_number_prior['factors_mean_var_ratio']),
                                                   shape=(self.n_comb, 1)))

        c2f_shape = self.factors_per_combs / self.n_fact
        self.comb2fact = pyro.sample('comb2fact',
                                     Gamma(alpha=c2f_shape,
                                           beta=self.factors_per_combs,
                                           shape=(self.n_comb, self.n_fact)))

        spot_factors_mu = self.combs_factors @ self.comb2fact
        spot_factors_sigma = torch.sqrt(self.combs_factors @ self.comb2fact / self.spot_fact_mean_var_ratio)

        self.spot_factors = pyro.sample('spot_factors',
                                        Gamma(mu=spot_factors_mu,
                                              sigma=spot_factors_sigma))

        # =====================Spot-specific additive component======================= #
        # molecule contribution that cannot be explained by cell state signatures
        # these counts are distributed between all genes not just expressed genes
        self.spot_add_hyp = pyro.sample('spot_add_hyp', Gamma(alpha=1, beta=1, shape=2))
        self.spot_add = pyro.sample('spot_add', Gamma(alpha=self.spot_add_hyp[0],
                                                      beta=self.spot_add_hyp[1],
                                                      shape=(self.n_cells, 1)))

        # =====================Gene-specific additive component ======================= #
        # per gene molecule contribution that cannot be explained by cell state signatures
        # these counts are distributed equally between all spots (e.g. background, free-floating RNA)
        self.gene_add_hyp = pyro.sample('gene_add_hyp', Gamma(alpha=1, beta=1, shape=2))
        self.gene_add = pyro.sample('gene_add', Gamma(alpha=self.gene_add_hyp[0],
                                                      beta=self.gene_add_hyp[1],
                                                      shape=(self.n_genes, 1)))

        # =====================Gene-specific overdispersion ======================= #
        self.phi_hyp = pyro.sample('phi_hyp',
                                   Gamma(mu=self.phi_hyp_prior['mean'],
                                         sigma=self.phi_hyp_prior['sd'],
                                         shape=(1, 1)))

        self.gene_E = pyro.sample('gene_E', dist.Exponential(torch.ones([self.n_genes, 1]) * self.phi_hyp[0, 0]))

        # =====================Expected expression ======================= #
        # expected expression
        self.mu_biol = torch.matmul(self.spot_factors[idx], self.gene_factors.T) * self.gene_level.T \
                       + self.gene_add.T + self.spot_add[idx]
        self.theta = torch.ones([1, 1]) / (self.gene_E.T * self.gene_E.T)
        # convert mean and overdispersion to total count and logits (input to NB)
        self.total_count, self.logits = _convert_mean_disp_to_counts_logits(self.mu_biol, self.theta,
                                                                            eps=1e-8)

        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        self.data_target = pyro.sample('data_target',
                                       dist.NegativeBinomial(total_count=self.total_count,
                                                             logits=self.logits),
                                       obs=x_data)

        # =====================Compute nUMI from each factor in spots  ======================= #
        nUMI = (self.spot_factors * (self.gene_factors * self.gene_level).sum(0))
        self.nUMI_factors = pyro.deterministic('nUMI_factors', nUMI)

    def plot_posterior_vs_dataV1(self):
        self.plot_posterior_vs_data(gene_fact_name='gene_factors',
                                    cell_fact_name='spot_factors_scaled')

    def plot_biol_spot_nUMI(self, fact_name='nUMI_factors'):
        plt.hist(np.log10(self.samples['post_sample_means'][fact_name].sum(1)), bins=50)
        plt.xlabel('Biological spot nUMI (log10)')
        plt.title('Biological spot nUMI')
        plt.tight_layout()

    def plot_spot_add(self):
        plt.hist(np.log10(self.samples['post_sample_means']['spot_add'][:, 0]), bins=50)
        plt.xlabel('UMI unexplained by biological factors')
        plt.title('Additive technical spot nUMI')
        plt.tight_layout()

    def plot_gene_E(self):
        plt.hist((self.samples['post_sample_means']['gene_E'][:, 0]), bins=50)
        plt.xlabel('E_g overdispersion parameter')
        plt.title('E_g overdispersion parameter')
        plt.tight_layout()

    def plot_gene_add(self):
        plt.hist((self.samples['post_sample_means']['gene_add'][:, 0]), bins=50)
        plt.xlabel('S_g additive background noise parameter')
        plt.title('S_g additive background noise parameter')
        plt.tight_layout()

    def plot_gene_level(self):
        plt.hist((self.samples['post_sample_means']['gene_level'][:, 0]), bins=50)
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
                   + self.samples['post_sample_means']['spot_add'])  # \

    # * self.samples['post_sample_means']['gene_E']

    def evaluate_stability(self, n_samples=1000, align=False):
        r""" Evaluate stability in factor contributions to spots.
        """

        self.b_evaluate_stability(node='spot_factors', n_samples=n_samples, align=align)

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

        return adata

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
