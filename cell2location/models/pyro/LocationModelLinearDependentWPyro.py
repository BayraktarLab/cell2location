########----------------########
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine

from cell2location.models.pyro.pyro_loc_model import PyroLocModel
from cell2location.distributions.Gamma import Gamma
from cell2location.distributions.NegativeBinomial import NegativeBinomial


def rand_tensor(shape, mean, sigma):
    r""" Helper for initializing variational parameters
    """
    return mean * torch.ones(shape) + sigma * torch.randn(shape)


########-------- defining the model itself - pyro -------- ########
class LocationModelLinearDependentWPyro(PyroLocModel):
    r"""Cell2location models the elements of :math:`D` as Negative Binomial distributed,
    given an unobserved rate :math:`mu` and a gene-specific over-dispersion parameter :math:`\alpha_g`
    which describes variance in expression of individual genes that is not explained by the regulatory programs:

    .. math::
        D_{s,g} \sim \mathtt{NB}(\mu_{s,g}, \alpha_g)

    The containment prior on overdispersion :math:`\alpha_g` parameter is used
    (for more details see: https://statmodeling.stat.columbia.edu/2018/04/03/justify-my-love/).

    The spatial expression levels of genes :math:`\mu_{s,g}` in the rate space are modelled
    as the sum of five non-negative components:

    .. math::
        \mu_{s,g} = m_{g} \left (\sum_{f} {w_{s,f} \: g_{f,g}} \right) + l_s + s_{g}

    Here, :math:`w_{s,f}` denotes regression weight of each program :math:`f` at location :math:`s` ;
    :math:`g_{f,g}` denotes the reference signatures of cell types :math:`f` of each gene :math:`g` - input to the model;
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
        Numpy array of gene expression (cols) in spatial locations (rows)
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
        super().__init__(cell_state_mat=cell_state_mat, X_data=X_data,
                         data_type=data_type, n_iter=n_iter,
                         learning_rate=learning_rate, total_grad_norm_constraint=total_grad_norm_constraint,
                         use_cuda=use_cuda, verbose=verbose, var_names=var_names, var_names_read=var_names_read,
                         obs_names=obs_names, fact_names=fact_names, sample_id=sample_id,
                         minibatch_size=minibatch_size, minibatch_seed=minibatch_seed)

        self.phi_hyp_prior = phi_hyp_prior
        self.n_comb = n_comb
        self.spot_fact_mean_var_ratio = spot_fact_mean_var_ratio

        cell_number_prior['factors_per_combs'] = (cell_number_prior['factors_per_spot'] /
                                                  cell_number_prior['combs_per_spot'])
        for k in cell_number_var_prior.keys():
            cell_number_prior[k] = cell_number_var_prior[k]
        self.cell_number_prior = cell_number_prior

        for k in gene_level_var_prior.keys():
            gene_level_prior[k] = gene_level_var_prior[k]
        self.gene_level_prior = gene_level_prior

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
            n_g_prior = self.n_var

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
                                            shape=(self.n_var, 1)))

        # scale cell state factors by gene_level
        self.gene_factors = pyro.deterministic('gene_factors', self.cell_state)

        # =====================Spot factors======================= #
        # prior on spot factors reflects the number of cells, fraction of their cytoplasm captured,
        # times heterogeneity in the total number of mRNA between individual cells with each cell type
        self.cells_per_spot = pyro.sample('cells_per_spot',
                                          Gamma(mu=self.cell_number_prior['cells_per_spot'],
                                                sigma=np.sqrt(self.cell_number_prior['cells_per_spot'] \
                                                              / self.cell_number_prior['cells_mean_var_ratio']),
                                                shape=(self.n_obs, 1)))

        self.comb_per_spot = pyro.sample('combs_per_spot',
                                         Gamma(mu=self.cell_number_prior['combs_per_spot'],
                                               sigma=np.sqrt(self.cell_number_prior['combs_per_spot'] \
                                                             / self.cell_number_prior['combs_mean_var_ratio']),
                                               shape=(self.n_obs, 1)))

        shape = self.comb_per_spot / self.n_comb
        rate = torch.ones([1, 1]) / self.cells_per_spot * self.comb_per_spot

        self.combs_factors = pyro.sample('combs_factors',
                                         Gamma(alpha=shape,
                                               beta=rate,
                                               shape=(self.n_obs, self.n_comb)))

        self.factors_per_combs = pyro.sample('factors_per_combs',
                                             Gamma(mu=self.cell_number_prior['factors_per_combs'],
                                                   sigma=np.sqrt(self.cell_number_prior['factors_per_combs'] \
                                                                 / self.cell_number_prior['factors_mean_var_ratio']),
                                                   shape=(self.n_comb, 1)))

        c2f_shape = self.factors_per_combs / torch.tensor(self.n_fact)
        self.comb2fact = pyro.sample('comb2fact',
                                     Gamma(alpha=c2f_shape,
                                           beta=self.factors_per_combs,
                                           shape=(self.n_comb, self.n_fact)))

        spot_factors_mu = self.combs_factors @ self.comb2fact
        spot_factors_sigma = torch.sqrt(spot_factors_mu / self.spot_fact_mean_var_ratio)

        self.spot_factors = pyro.sample('spot_factors',
                                        Gamma(mu=spot_factors_mu,
                                              sigma=spot_factors_sigma))

        # =====================Spot-specific additive component======================= #
        # molecule contribution that cannot be explained by cell state signatures
        # these counts are distributed between all genes not just expressed genes
        self.spot_add_hyp = pyro.sample('spot_add_hyp', Gamma(alpha=1, beta=1, shape=2))
        self.spot_add = pyro.sample('spot_add', Gamma(alpha=self.spot_add_hyp[0],
                                                      beta=self.spot_add_hyp[1],
                                                      shape=(self.n_obs, 1)))

        # =====================Gene-specific additive component ======================= #
        # per gene molecule contribution that cannot be explained by cell state signatures
        # these counts are distributed equally between all spots (e.g. background, free-floating RNA)
        self.gene_add_hyp = pyro.sample('gene_add_hyp', Gamma(alpha=1, beta=1, shape=2))
        self.gene_add = pyro.sample('gene_add', Gamma(alpha=self.gene_add_hyp[0],
                                                      beta=self.gene_add_hyp[1],
                                                      shape=(self.n_var, 1)))

        # =====================Gene-specific overdispersion ======================= #
        self.phi_hyp = pyro.sample('phi_hyp',
                                   Gamma(mu=self.phi_hyp_prior['mean'],
                                         sigma=self.phi_hyp_prior['sd'],
                                         shape=(1, 1)))

        self.gene_E = pyro.sample('gene_E', dist.Exponential(torch.ones([self.n_var, 1]) * self.phi_hyp[0, 0]))

        # =====================Expected expression ======================= #
        # expected expression
        self.mu_biol = torch.matmul(self.spot_factors[idx], self.gene_factors.T) * self.gene_level.T \
                       + self.gene_add.T + self.spot_add[idx]
        self.theta = torch.ones([1, 1]) / (self.gene_E.T * self.gene_E.T)

        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        self.data_target = pyro.sample('data_target',
                                       NegativeBinomial(mu=self.mu_biol,
                                                        theta=self.theta),
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
        self.alpha = 1 / (self.samples['post_sample_means']['gene_E'].T * self.samples['post_sample_means']['gene_E'].T)

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
