from typing import Optional

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from einops import rearrange
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr
from pyro.nn import PyroModule
from scipy.sparse import coo_matrix
from scvi import REGISTRY_KEYS
from scvi.nn import one_hot

from cell2location.nn.CellCommunicationToEffectNN import CellCommunicationToTfActivityNN


class CellCommModule(PyroModule):
    r"""
    Cell2location models the elements of :math:`D` as Negative Binomial distributed,
    given an unobserved gene expression level (rate) :math:`mu` and a gene- and batch-specific
    over-dispersion parameter :math:`\alpha_{e,g}` which accounts for unexplained variance:

    .. math::
        D_{s,g} \sim \mathtt{NB}(\mu_{s,g}, \alpha_{e,g})

    Here, :math:`w_{s,f}` denotes regression weight of each reference signature :math:`f` at location :math:`s`, which can be interpreted as the expected number of cells at location :math:`s` that express reference signature :math:`f`;
    :math:`g_{f,g}` denotes the reference signatures of cell types :math:`f` of each gene :math:`g`, `cell_state_df` input ;
    """
    n_pathways = 150
    use_pathway_interaction_effect = True
    dropout_rate = 0.0
    use_non_negative_weights = False
    min_distance = 25.0
    r_l_affinity_alpha_prior = 10.0
    use_global_cell_abundance_model = False
    record_sr_occupancy = False
    use_spatial_receptor_info_remove_sp_signal = True

    def __init__(
        self,
        n_obs,
        n_vars,
        n_factors,
        n_batch,
        detection_mean=1 / 2,
        detection_alpha=20.0,
        m_g_gene_level_prior={"mean": 1, "mean_var_ratio": 1.0, "alpha_mean": 3.0},
        alpha_g_phi_hyp_prior={"alpha": 9.0, "beta": 3.0},
        detection_hyp_prior={"mean_alpha": 10.0},
        w_sf_mean_var_ratio=5.0,
        init_vals: Optional[dict] = None,
        init_alpha: float = 20.0,
        dropout_p: float = 0.0,
        receptor_abundance: Optional[np.ndarray] = None,
        use_spatial_receptor_info: bool = False,
        per_cell_type_normalisation: Optional[np.ndarray] = None,
        signal_receptor_mask: Optional[np.ndarray] = None,
        receptor_tf_mask: Optional[np.ndarray] = None,
        use_learnable_mean_var_ratio: bool = False,
        average_distance_prior: float = 50.0,
        distances: Optional[coo_matrix] = None,
        n_hidden: int = 256,
        use_cell_abundance_normalisation: bool = True,
        use_alpha_likelihood: bool = True,
        use_normal_likelihood: bool = False,
        fixed_w_sf_mean_var_ratio: Optional[float] = None,
    ):
        super().__init__()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_factors = n_factors
        self.n_batch = n_batch
        self.n_hidden = n_hidden

        self.m_g_gene_level_prior = m_g_gene_level_prior

        self.alpha_g_phi_hyp_prior = alpha_g_phi_hyp_prior
        self.w_sf_mean_var_ratio = w_sf_mean_var_ratio
        detection_hyp_prior["mean"] = detection_mean
        detection_hyp_prior["alpha"] = detection_alpha
        self.detection_hyp_prior = detection_hyp_prior

        self.dropout_p = dropout_p
        if self.dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=self.dropout_p)

        if receptor_abundance is not None:
            self.register_buffer("receptor_abundance", torch.tensor(receptor_abundance.astype("float32")))
        self.use_spatial_receptor_info = use_spatial_receptor_info
        if per_cell_type_normalisation is not None:
            self.register_buffer(
                "per_cell_type_normalisation", torch.tensor(per_cell_type_normalisation.astype("float32"))
            )
        self.signal_receptor_mask = signal_receptor_mask
        self.receptor_tf_mask = receptor_tf_mask

        self.use_learnable_mean_var_ratio = use_learnable_mean_var_ratio
        self.average_distance_prior = average_distance_prior
        if distances is not None:
            distances = coo_matrix(distances).astype("float32")
            self.distances_scipy = distances
            self.register_buffer(
                "distances",
                torch.sparse_coo_tensor(
                    torch.tensor(np.array([distances.row, distances.col])),
                    torch.tensor(distances.data.astype("float32")),
                    distances.shape,
                ),
            )
        self.use_cell_abundance_normalisation = use_cell_abundance_normalisation
        self.use_alpha_likelihood = use_alpha_likelihood
        self.use_normal_likelihood = use_normal_likelihood
        self.fixed_w_sf_mean_var_ratio = fixed_w_sf_mean_var_ratio

        self.weights = PyroModule()

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))
            self.init_alpha = init_alpha
            self.register_buffer("init_alpha_tt", torch.tensor(self.init_alpha))

        self.register_buffer(
            "detection_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["mean_alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_beta",
            torch.tensor(self.detection_hyp_prior["mean_alpha"] / self.detection_hyp_prior["mean"]),
        )

        # compute hyperparameters from mean and sd
        self.register_buffer("m_g_mu_hyp", torch.tensor(self.m_g_gene_level_prior["mean"]))
        self.register_buffer(
            "m_g_mu_mean_var_ratio_hyp",
            torch.tensor(self.m_g_gene_level_prior["mean_var_ratio"]),
        )

        self.register_buffer("m_g_alpha_hyp_mean", torch.tensor(self.m_g_gene_level_prior["alpha_mean"]))

        self.register_buffer(
            "alpha_g_phi_hyp_prior_alpha",
            torch.tensor(self.alpha_g_phi_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "alpha_g_phi_hyp_prior_beta",
            torch.tensor(self.alpha_g_phi_hyp_prior["beta"]),
        )

        self.register_buffer("w_sf_mean_var_ratio_tensor", torch.tensor(self.w_sf_mean_var_ratio))

        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("ones_1d", torch.ones(1))
        self.register_buffer("zeros", torch.zeros((1, 1)))
        self.register_buffer("ones_1_n_factors", torch.ones((1, self.n_factors)))
        self.register_buffer("ones_n_batch_1", torch.ones((self.n_batch, 1)))
        self.register_buffer("eps", torch.tensor(1e-8))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        signal_abundance = tensor_dict["signal_abundance"]
        w_sf = tensor_dict["w_sf"]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        kwargs = {}
        if "positions" in tensor_dict.keys():
            kwargs["positions"] = tensor_dict["positions"]
        if "tiles" in tensor_dict.keys():
            kwargs["tiles"] = tensor_dict["tiles"]
        if "tiles_unexpanded" in tensor_dict.keys():
            kwargs["tiles_unexpanded"] = tensor_dict["tiles_unexpanded"]
        if "in_tissue" in tensor_dict.keys():
            kwargs["in_tissue"] = tensor_dict["in_tissue"].bool()
        if "w_sf_lvl2" in tensor_dict.keys():
            kwargs["w_sf_lvl2"] = tensor_dict["w_sf_lvl2"]
        return (signal_abundance, w_sf, ind_x, batch_index), kwargs

    def create_plates(
        self,
        signal_abundance,
        w_sf,
        idx,
        batch_index,
        tiles: torch.Tensor = None,
        tiles_unexpanded: torch.Tensor = None,
        positions: torch.Tensor = None,
        in_tissue: torch.Tensor = None,
        w_sf_lvl2: torch.Tensor = None,
    ):
        if tiles_unexpanded is not None:
            tiles_in_use = (tiles.mean(0) > torch.tensor(0.99, device=tiles.device)).bool()
            obs_in_use = (tiles_unexpanded[:, tiles_in_use].sum(1) > torch.tensor(0.0, device=tiles.device)).bool()
            idx = idx[obs_in_use]
        return pyro.plate("obs_plate", size=self.n_obs, dim=-2, subsample=idx)

    def list_obs_plate_vars(self):
        """
        Create a dictionary with:

        1. "name" - the name of observation/minibatch plate;
        2. "input" - indexes of model args to provide to encoder network when using amortised inference;
        3. "sites" - dictionary with

          * keys - names of variables that belong to the observation plate (used to recognise
            and merge posterior samples for minibatch variables)
          * values - the dimensions in non-plate axis of each variable (used to construct output
            layer of encoder network when using amortised inference)
        """
        input_transform = torch.log1p
        n_in = self.n_vars

        return {
            "name": "obs_plate",
            "input": [0, 2],  # expression data + (optional) batch index
            "n_in": n_in,
            "input_transform": [
                input_transform,
                lambda x: x,
            ],  # how to transform input data before passing to NN
            "input_normalisation": [
                False,
                False,
            ],  # whether to normalise input data before passing to NN
            "sites": {},
        }

    def get_cell_communication_module(
        self,
        name,
        output_transform="softplus",
        n_out: int = 1,
        average_distance_prior: float = 50.0,
    ):
        # create module if it doesn't exist
        if getattr(self.weights, name, None) is None:
            deep_setattr(
                self.weights,
                name,
                CellCommunicationToTfActivityNN(
                    name=name,
                    mode="signal_receptor_tf_effect_spatial",
                    output_transform=output_transform,
                    n_tfs=self.n_factors,
                    n_signals=self.signal_receptor_mask.shape[0],
                    n_receptors=self.signal_receptor_mask.shape[1],
                    n_out=n_out,
                    n_pathways=self.n_pathways,
                    signal_receptor_mask=self.signal_receptor_mask,  # tells which receptors can bind which ligands
                    receptor_tf_mask=self.receptor_tf_mask,  # tells which receptors can influence which TF (eg nuclear receptor = TF)
                    dropout_rate=self.dropout_rate,
                    use_horseshoe_prior=True,
                    use_gamma_horseshoe_prior=False,
                    weights_prior_tau=1.0,
                    use_pathway_interaction_effect=self.use_pathway_interaction_effect,
                    average_distance_prior=average_distance_prior,
                    use_non_negative_weights=self.use_non_negative_weights,
                    r_l_affinity_alpha_prior=self.r_l_affinity_alpha_prior,
                    use_global_cell_abundance_model=self.use_global_cell_abundance_model,
                ),
            )
        # get module
        return deep_getattr(self.weights, name)

    def cell_comm_effect(
        self,
        signal_abundance,
        receptor_abundance,
        distances,
        tiles,
        obs_plate,
        average_distance_prior=50.0,
    ):
        # get module
        module = self.get_cell_communication_module(
            name="lr2abundance",
            output_transform="softplus",
            n_out=1,
            average_distance_prior=average_distance_prior,
        )
        # compute LR occupancy
        bound_receptor_abundance_src = module.signal_receptor_occupancy_spatial(
            signal_abundance,
            receptor_abundance,
            distances,
            tiles,
            obs_plate,
        )
        # compute cell abundance prediction
        w_sf_mu = module.signal_receptor_tf_effect_spatial(
            bound_receptor_abundance_src,
        )
        return w_sf_mu, bound_receptor_abundance_src

    def forward(
        self,
        signal_abundance,
        w_sf,
        idx,
        batch_index,
        tiles: torch.Tensor = None,
        tiles_unexpanded: torch.Tensor = None,
        positions: torch.Tensor = None,
        in_tissue: torch.Tensor = None,
        w_sf_lvl2: torch.Tensor = None,
    ):
        obs_plate = self.create_plates(
            signal_abundance=signal_abundance,
            w_sf=w_sf,
            idx=idx,
            batch_index=batch_index,
            tiles=tiles,
            tiles_unexpanded=tiles_unexpanded,
            positions=positions,
            in_tissue=in_tissue,
        )
        if tiles_unexpanded is not None:
            tiles_in_use = (tiles.mean(0) > torch.tensor(0.99, device=tiles.device)).bool()
            obs_in_use = (tiles_unexpanded[:, tiles_in_use].sum(1) > torch.tensor(0.0, device=tiles.device)).bool()
            batch_index = batch_index[obs_in_use]
        obs2sample = one_hot(batch_index, self.n_batch).float()

        if getattr(self, "distances", None) is not None:
            distances = self.distances
        elif positions is not None:
            # compute distance using positions [observations, 2]
            distances = (
                (positions.unsqueeze(1) - positions.unsqueeze(0))  # [observations, 1, 2]  # [1, observations, 2]
                .pow(2)
                .sum(-1)
                .sqrt()
            ) + torch.tensor(self.min_distance, device=positions.device)

        # =====================Cell abundances w_sf======================= #
        if self.use_spatial_receptor_info:
            if self.use_spatial_receptor_info_remove_sp_signal:
                receptor_abundance_ = torch.einsum(
                    "fr,cf,f -> cr",
                    self.receptor_abundance.T,
                    w_sf,
                    self.per_cell_type_normalisation,
                )
                receptor_abundance_norm = torch.einsum(
                    "fr,r -> fr",
                    self.receptor_abundance.T,
                    self.receptor_abundance.sum(-1),
                )
                receptor_abundance = torch.einsum(
                    "fr,cr -> cfr",
                    receptor_abundance_norm,
                    receptor_abundance_,
                )
            else:
                receptor_abundance = torch.einsum(
                    "fr,cf,f -> cfr",
                    self.receptor_abundance.T,
                    w_sf,
                    self.per_cell_type_normalisation,
                )
        else:
            receptor_abundance = self.receptor_abundance.T
        w_sf_mu_cell_comm, bound_receptor_abundance_src = self.cell_comm_effect(
            signal_abundance=signal_abundance,
            receptor_abundance=receptor_abundance,
            distances=distances,
            tiles=tiles,
            average_distance_prior=self.average_distance_prior,
            obs_plate=obs_plate,
        )
        if not self.training and self.record_sr_occupancy:
            with obs_plate:
                # {sr pair, location * cell type} -> {sr pair, location, cell type}
                bound_receptor_abundance_src = rearrange(
                    bound_receptor_abundance_src,
                    "r (c f) -> r c f",
                    f=self.receptor_abundance.shape[-1],
                )
                if tiles_unexpanded is not None:
                    bound_receptor_abundance_src = bound_receptor_abundance_src[:, obs_in_use, :]
                pyro.deterministic(
                    "bound_receptor_abundance_sr_c_f",
                    bound_receptor_abundance_src,
                )
        if self.fixed_w_sf_mean_var_ratio is not None:
            w_sf_mean_var_ratio = torch.tensor(self.fixed_w_sf_mean_var_ratio, device=w_sf_mu_cell_comm.device)
        else:
            w_sf_mean_var_ratio_hyp = pyro.sample(
                "w_sf_mean_var_ratio_hyp_lik",
                dist.Gamma(self.w_sf_mean_var_ratio_tensor, self.ones).expand([1, 1]).to_event(2),
            )  # prior mean 5.0
            w_sf_mean_var_ratio = pyro.sample(
                "w_sf_mean_var_ratio_lik",
                dist.Exponential(w_sf_mean_var_ratio_hyp).expand([1, self.n_factors]).to_event(2),
            )  # (self.n_batch, self.n_vars) prior mean 0.2
            if self.use_normal_likelihood:
                w_sf_mean_var_ratio = w_sf_mean_var_ratio + torch.tensor(1.0 / 50.0, device=w_sf_mean_var_ratio.device)
            else:
                w_sf_mean_var_ratio = self.ones / (
                    w_sf_mean_var_ratio + torch.tensor(1.0 / 50.0, device=w_sf_mean_var_ratio.device)
                ) + torch.tensor(5.0, device=w_sf_mean_var_ratio.device)
        if tiles_unexpanded is not None:
            w_sf_mu_cell_comm = w_sf_mu_cell_comm[obs_in_use]
            w_sf = w_sf[obs_in_use]
            if w_sf_lvl2 is not None:
                w_sf_lvl2 = w_sf_lvl2[obs_in_use]
        if self.use_cell_abundance_normalisation:
            # =====================Location-specific detection efficiency ======================= #
            # y_s with hierarchical mean prior
            detection_mean_y_e = pyro.sample(
                "detection_mean_y_e",
                dist.Gamma(
                    self.ones * self.detection_mean_hyp_prior_alpha,
                    self.ones * self.detection_mean_hyp_prior_beta,
                )
                .expand([self.n_batch, 1])
                .to_event(2),
            )
            detection_hyp_prior_alpha = pyro.deterministic(
                "detection_hyp_prior_alpha",
                self.ones_n_batch_1 * self.detection_hyp_prior_alpha,
            )

            beta = (obs2sample @ detection_hyp_prior_alpha) / (obs2sample @ detection_mean_y_e)
            with obs_plate:
                k = "detection_y_s"
                detection_y_s = pyro.sample(
                    k,
                    dist.Gamma(obs2sample @ detection_hyp_prior_alpha, beta),
                )  # (self.n_obs, 1)
            w_sf_mu_cell_comm = w_sf_mu_cell_comm * detection_y_s

        if w_sf_lvl2 is not None:
            with obs_plate:
                pyro.deterministic("w_sf_wo_lvl2_cell_comm", w_sf_mu_cell_comm)  # (self.n_obs, self.n_factors)
            w_sf_mu_cell_comm = w_sf_mu_cell_comm * w_sf_lvl2

        with obs_plate:
            k = "w_sf"
            pyro.deterministic(f"{k}_cell_comm", w_sf_mu_cell_comm)  # (self.n_obs, self.n_factors)

        if self.use_alpha_likelihood:
            with obs_plate:
                pyro.sample(
                    f"{k}_obs",
                    dist.Gamma(
                        w_sf_mean_var_ratio,
                        w_sf_mean_var_ratio / w_sf_mu_cell_comm,
                    ),
                    obs=w_sf,
                )  # (self.n_obs, self.n_factors)
        elif self.use_normal_likelihood:
            with obs_plate:
                pyro.sample(
                    f"{k}_obs",
                    dist.Normal(
                        w_sf_mu_cell_comm,
                        w_sf_mean_var_ratio,
                    ),
                    obs=w_sf,
                )  # (self.n_obs, self.n_factors)
        else:
            with obs_plate:
                pyro.sample(
                    f"{k}_obs",
                    dist.Gamma(
                        w_sf_mu_cell_comm * w_sf_mean_var_ratio,
                        w_sf_mean_var_ratio,
                    ),
                    obs=w_sf,
                )  # (self.n_obs, self.n_factors)
