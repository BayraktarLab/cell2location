from typing import Optional

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from einops import rearrange
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr
from pyro.nn import PyroModule
from scipy.sparse import coo_matrix, csr_matrix
from scvi import REGISTRY_KEYS
from scvi.nn import one_hot

from cell2location.nn.CellCommunicationToEffectNN import CellCommunicationToTfActivityNN


class LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel(PyroModule):
    r"""
    Cell2location models the elements of :math:`D` as Negative Binomial distributed,
    given an unobserved gene expression level (rate) :math:`mu` and a gene- and batch-specific
    over-dispersion parameter :math:`\alpha_{e,g}` which accounts for unexplained variance:

    .. math::
        D_{s,g} \sim \mathtt{NB}(\mu_{s,g}, \alpha_{e,g})

    The expression level of genes :math:`\mu_{s,g}` in the mRNA count space is modelled
    as a linear function of expression signatures of reference cell types :math:`g_{f,g}`:

    .. math::
        \mu_{s,g} = (m_{g} \left (\sum_{f} {w_{s,f} \: g_{f,g}} \right) + s_{e,g}) y_{s}

    Here, :math:`w_{s,f}` denotes regression weight of each reference signature :math:`f` at location :math:`s`, which can be interpreted as the expected number of cells at location :math:`s` that express reference signature :math:`f`;
    :math:`g_{f,g}` denotes the reference signatures of cell types :math:`f` of each gene :math:`g`, `cell_state_df` input ;
    :math:`m_{g}` denotes a gene-specific scaling parameter which adjusts for global differences in sensitivity between technologies (platform effect);
    :math:`y_{s}` denotes a location/observation-specific scaling parameter which adjusts for differences in sensitivity between observations and batches;
    :math:`s_{e,g}` is additive component that account for gene- and location-specific shift, such as due to contaminating or free-floating RNA.

    To account for the similarity of location patterns across cell types, :math:`w_{s,f}` is modelled using
    another layer  of decomposition (factorization) using :math:`r={1, .., R}` groups of cell types,
    that can be interpreted as cellular compartments or tissue zones. Unless stated otherwise, R is set to 50.

    Corresponding graphical model can be found in supplementary methods:
    https://www.biorxiv.org/content/10.1101/2020.11.15.378125v1.supplementary-material

    Approximate Variational Inference is used to estimate the posterior distribution of all model parameters.

    Estimation of absolute cell abundance :math:`w_{s,f}` is guided using informed prior on the number of cells
    (argument called `N_cells_per_location`). It is a tissue-level global estimate, which can be derived from histology
    images (H&E or DAPI), ideally paired to the spatial expression data or at least representing the same tissue type.
    This parameter can be estimated by manually counting nuclei in a 10-20 locations in the histology image
    (e.g. using 10X Loupe browser), and computing the average cell abundance.
    An appropriate setting of this prior is essential to inform the estimation of absolute cell type abundance values,
    however, the model is robust to a range of similar values.
    In settings where suitable histology images are not available, the size of capture regions relative to
    the expected size of cells can be used to estimate `N_cells_per_location`.
    `N_cells_per_location` has to be a scalar or an array of shape (n_obs, 1).

    The prior on detection efficiency per location :math:`y_s` is selected to discourage over-normalisation, such that
    unless data has evidence of strong technical effect, the effect is assumed to be small and close to
    the mean sensitivity for each batch :math:`y_e`:

    .. math::
        y_s \sim Gamma(detection\_alpha, detection\_alpha / y_e)

    where y_e is unknown/latent average detection efficiency in each batch/experiment:

    .. math::
        y_e \sim Gamma(10, 10 / detection\_mean)

    """

    # training mode without observed data (just using priors)
    training_wo_observed = False
    training_wo_initial = False

    n_cell_compartments = 3
    named_dims = {
        "cell_compartment_w_sfk": -3,
    }
    n_tiles = 1
    use_concatenated_cnn = False

    n_pathways = 8
    use_pathway_interaction_effect = True
    dropout_rate = 0.0
    use_non_negative_weights = False

    def __init__(
        self,
        n_obs,
        n_vars,
        n_factors,
        n_batch,
        cell_state_mat,
        n_groups: int = 50,
        detection_mean=1 / 2,
        detection_alpha=20.0,
        m_g_gene_level_prior={"mean": 1, "mean_var_ratio": 1.0, "alpha_mean": 3.0},
        N_cells_per_location: float = 8.0,  # float or array
        A_factors_per_location: float = 7.0,
        B_groups_per_location: float = 7.0,
        N_cells_mean_var_ratio: float = 1.0,
        N_cells_per_location_alpha_prior: float = None,
        A_B_per_location_alpha_prior: float = None,
        alpha_g_phi_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={
            "alpha": 1.0,
            "beta": 100.0,
        },
        detection_hyp_prior={"mean_alpha": 10.0},
        detection_cell_type_prior_alpha=10.0,
        use_per_cell_type_normalisation: bool = False,
        w_sf_mean_var_ratio=5.0,
        init_vals: Optional[dict] = None,
        init_alpha: float = 20.0,
        dropout_p: float = 0.0,
        use_cell_comm_prior_on_w_sf: bool = False,
        use_cell_comm_likelihood_w_sf: bool = False,
        signal_bool: Optional[np.ndarray] = None,
        receptor_bool: Optional[np.ndarray] = None,
        receptor_bool_b: Optional[np.ndarray] = None,
        signal_receptor_mask: Optional[np.ndarray] = None,
        receptor_tf_mask: Optional[np.ndarray] = None,
        use_learnable_mean_var_ratio: bool = False,
        use_independent_prior_on_w_sf: bool = False,
        use_proportion_factorisation_prior_on_w_sf: bool = False,
        average_distance_prior: float = 50.0,
        distances: Optional[coo_matrix] = None,
        sliding_window_size: Optional[int] = 0,
        amortised_sliding_window_size: Optional[int] = 0,
        sliding_window_size_list: Optional[list] = None,
        use_normalising_factor_y_s: bool = False,
        image_size: Optional[tuple] = None,
        use_aggregated_w_sf: bool = False,
        use_aggregated_detection_y_s: bool = False,
        use_cell_compartments: bool = False,
        use_weigted_cnn_weights: bool = False,
        n_hidden: int = 256,
    ):
        super().__init__()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_factors = n_factors
        self.n_batch = n_batch
        self.n_groups = n_groups
        self.n_hidden = n_hidden

        self.m_g_gene_level_prior = m_g_gene_level_prior

        self.alpha_g_phi_hyp_prior = alpha_g_phi_hyp_prior
        self.w_sf_mean_var_ratio = w_sf_mean_var_ratio
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior
        detection_hyp_prior["mean"] = detection_mean
        detection_hyp_prior["alpha"] = detection_alpha
        self.detection_hyp_prior = detection_hyp_prior

        self.dropout_p = dropout_p
        if self.dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=self.dropout_p)

        self.use_cell_comm_prior_on_w_sf = use_cell_comm_prior_on_w_sf
        self.use_cell_comm_likelihood_w_sf = use_cell_comm_likelihood_w_sf
        if signal_bool is not None:
            self.register_buffer("signal_bool", torch.tensor(signal_bool.astype("int32")))
        if receptor_bool is not None:
            self.register_buffer("receptor_bool", torch.tensor(receptor_bool.astype("int32")))
            if receptor_bool_b is None:
                raise ValueError("receptor_bool_b must be provided if receptor_bool is provided")
            self.register_buffer("receptor_bool_b", torch.tensor(receptor_bool_b.astype("int32")))
        self.signal_receptor_mask = signal_receptor_mask
        self.receptor_tf_mask = receptor_tf_mask

        self.use_learnable_mean_var_ratio = use_learnable_mean_var_ratio
        self.use_independent_prior_on_w_sf = use_independent_prior_on_w_sf
        self.use_proportion_factorisation_prior_on_w_sf = use_proportion_factorisation_prior_on_w_sf
        self.average_distance_prior = average_distance_prior
        if distances is not None:
            distances = coo_matrix(distances)
            self.distances_scipy = distances
            self.register_buffer(
                "distances",
                torch.sparse_coo_tensor(
                    torch.tensor(np.array([distances.row, distances.col])),
                    torch.tensor(distances.data.astype("float32")),
                    distances.shape,
                ),
            )
        self.sliding_window_size = (
            sliding_window_size if sliding_window_size_list is None else max(sliding_window_size_list)
        )
        self.amortised_sliding_window_size = amortised_sliding_window_size
        self.sliding_window_size_list_exist = sliding_window_size_list is not None
        self.sliding_window_size_list = np.array(sliding_window_size_list)
        self.use_normalising_factor_y_s = use_normalising_factor_y_s
        self.image_size = image_size
        self.use_aggregated_w_sf = use_aggregated_w_sf
        self.use_aggregated_detection_y_s = use_aggregated_detection_y_s
        self.use_cell_compartments = use_cell_compartments
        self.use_weigted_cnn_weights = use_weigted_cnn_weights

        self.weights = PyroModule()

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))
            self.init_alpha = init_alpha
            self.register_buffer("init_alpha_tt", torch.tensor(self.init_alpha))

        factors_per_groups = A_factors_per_location / B_groups_per_location

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
        if use_per_cell_type_normalisation:
            self.register_buffer(
                "detection_cell_type_prior_alpha",
                torch.tensor(detection_cell_type_prior_alpha),
            )
        self.use_per_cell_type_normalisation = use_per_cell_type_normalisation

        # compute hyperparameters from mean and sd
        self.register_buffer("m_g_mu_hyp", torch.tensor(self.m_g_gene_level_prior["mean"]))
        self.register_buffer(
            "m_g_mu_mean_var_ratio_hyp",
            torch.tensor(self.m_g_gene_level_prior["mean_var_ratio"]),
        )

        self.register_buffer("m_g_alpha_hyp_mean", torch.tensor(self.m_g_gene_level_prior["alpha_mean"]))

        self.cell_state_mat = cell_state_mat
        self.register_buffer("cell_state", torch.tensor(cell_state_mat.T))

        if isinstance(N_cells_per_location, np.ndarray):
            assert (
                N_cells_per_location.shape[0] == self.n_obs
            ), "N_cells_per_location must have shape (n_obs, 1) or be a scalar"
        self.register_buffer("N_cells_per_location", torch.tensor(N_cells_per_location))
        self.register_buffer("A_factors_per_location", torch.tensor(A_factors_per_location))
        self.register_buffer("factors_per_groups", torch.tensor(factors_per_groups))
        self.register_buffer("B_groups_per_location", torch.tensor(B_groups_per_location))
        assert (N_cells_per_location_alpha_prior is None) or (
            N_cells_mean_var_ratio is None
        ), "N_cells_per_location_alpha_prior and N_cells_mean_var_ratio cannot be provided at the same time"
        if N_cells_per_location_alpha_prior is not None:
            self.register_buffer(
                "N_cells_per_location_alpha_prior", torch.tensor(float(N_cells_per_location_alpha_prior))
            )
            self.N_cells_mean_var_ratio = None
        else:
            self.register_buffer("N_cells_mean_var_ratio", torch.tensor(float(N_cells_mean_var_ratio)))
            self.N_cells_per_location_alpha_prior = None
        if A_B_per_location_alpha_prior is not None:
            self.register_buffer("A_B_per_location_alpha_prior", torch.tensor(float(A_B_per_location_alpha_prior)))
        else:
            self.A_B_per_location_alpha_prior = None

        self.register_buffer(
            "alpha_g_phi_hyp_prior_alpha",
            torch.tensor(self.alpha_g_phi_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "alpha_g_phi_hyp_prior_beta",
            torch.tensor(self.alpha_g_phi_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_alpha",
            torch.tensor(self.gene_add_alpha_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_beta",
            torch.tensor(self.gene_add_alpha_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_alpha",
            torch.tensor(self.gene_add_mean_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_beta",
            torch.tensor(self.gene_add_mean_hyp_prior["beta"]),
        )

        self.register_buffer("w_sf_mean_var_ratio_tensor", torch.tensor(self.w_sf_mean_var_ratio))

        self.register_buffer("n_factors_tensor", torch.tensor(self.n_factors))
        self.register_buffer("n_groups_tensor", torch.tensor(self.n_groups))

        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("ones_1d", torch.ones(1))
        self.register_buffer("zeros", torch.zeros((1, 1)))
        self.register_buffer("ones_1_n_groups", torch.ones((1, self.n_groups)))
        self.register_buffer("ones_1_n_factors", torch.ones((1, self.n_factors)))
        self.register_buffer("ones_n_batch_1", torch.ones((self.n_batch, 1)))
        self.register_buffer("eps", torch.tensor(1e-8))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x_data = tensor_dict[REGISTRY_KEYS.X_KEY]
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
        if "normalising_factor_y_s" in tensor_dict.keys():
            kwargs["normalising_factor_y_s"] = tensor_dict["normalising_factor_y_s"]
        if "x_data_normalised" in tensor_dict.keys():
            kwargs["x_data_normalised"] = tensor_dict["x_data_normalised"]
        return (x_data, ind_x, batch_index), kwargs

    def create_plates(
        self,
        x_data,
        idx,
        batch_index,
        tiles: torch.Tensor = None,
        tiles_unexpanded: torch.Tensor = None,
        positions: torch.Tensor = None,
        in_tissue: torch.Tensor = None,
        normalising_factor_y_s: Optional[torch.Tensor] = None,
        x_data_normalised: Optional[torch.Tensor] = None,
    ):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-2, subsample=idx)

    def conv2d_aggregate(self, x_data):
        x_data_agg = self.aggregate_conv2d(
            x_data,
            n_tiles=self.n_tiles,
            size=max(self.amortised_sliding_window_size, self.sliding_window_size),
            padding="same",
        )
        x_data = torch.cat([x_data, x_data_agg], dim=-1)
        return torch.log1p(x_data)

    def learnable_conv2d(self, x_data):
        x_data = torch.log1p(x_data)
        x_data_agg = self.learnable_neighbour_effect_conv2d_nn_layers(
            x_data,
            n_tiles=self.n_tiles,
            name="amortised_sliding_window",
            size=max(self.amortised_sliding_window_size, self.sliding_window_size),
            n_out=self.n_hidden,
            padding="same",
        )
        # x_data = self.aggregate_conv2d(x_data, padding="same")
        if self.use_concatenated_cnn:
            x_data_agg = torch.cat([x_data, x_data_agg], dim=-1)
        return x_data_agg

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

        if (self.amortised_sliding_window_size > 0) and (self.sliding_window_size == 0):
            input_transform = self.learnable_conv2d
            if self.use_concatenated_cnn:
                n_in = self.n_vars + self.n_hidden
            else:
                n_in = self.n_hidden
        elif (self.amortised_sliding_window_size == 0) and (self.sliding_window_size > 0):
            input_transform = self.conv2d_aggregate
            n_in = self.n_vars * 2
        elif (self.amortised_sliding_window_size > 0) and (self.sliding_window_size > 0):
            input_transform = self.learnable_conv2d
            if self.use_concatenated_cnn:
                n_in = self.n_vars + self.n_hidden
            else:
                n_in = self.n_hidden
        input = [0, 2]  # expression data + (optional) batch index
        if self.use_normalising_factor_y_s:
            input = ["x_data_normalised", 2]
        return {
            "name": "obs_plate",
            "input": input,  # expression data + (optional) batch index
            "n_in": n_in,
            "input_transform": [
                input_transform,
                lambda x: x,
            ],  # how to transform input data before passing to NN
            "input_normalisation": [
                False,
                False,
            ],  # whether to normalise input data before passing to NN
            "sites": {
                "n_s_cells_per_location": 1,
                "b_s_groups_per_location": 1,
                "a_s_factors_per_location": 1,
                "z_sr_groups_factors": self.n_groups,
                "w_sf": self.n_factors,
                "prior_w_sf": self.n_factors,
                "cell_compartment_w_sfk": (self.n_factors, int(self.n_cell_compartments - 1)),
                "detection_y_s": 1,
            },
        }

    def reshape_input_2d(self, x, n_tiles=1, axis=-2, axis_offset=-4):
        # conv2d expects 4d input: [batch, channels, height, width]
        if self.image_size is None:
            sizex = sizey = int(np.sqrt(x.shape[axis] / n_tiles))
        else:
            sizex, sizey = self.image_size
        # here batch dim has just one element
        if n_tiles > 1:
            return rearrange(x, "(t p o) g -> t g p o", p=sizex, o=sizey, t=n_tiles)
        else:
            return rearrange(x, "(p o) g -> g p o", p=sizex, o=sizey).unsqueeze(axis_offset)

    def reshape_input_2d_inverse(self, x, n_tiles=1, axis=-2, axis_offset=-4):
        # conv2d expects 4d input: [batch, channels, height, width]
        # here batch dim has just one element
        if n_tiles > 1:
            return rearrange(x.squeeze(axis_offset), "t g p o -> (t p o) g")
        else:
            return rearrange(x.squeeze(axis_offset), "g p o -> (p o) g")

    def crop_according_to_valid_padding(self, x, n_tiles=1):
        # remove observations that will not be included after convolution with padding='valid'
        # reshape to 2d
        x = self.reshape_input_2d(x, n_tiles=n_tiles)
        # crop to valid observations
        indx = np.arange(self.sliding_window_size // 2, x.shape[-2] - (self.sliding_window_size // 2))
        indy = np.arange(self.sliding_window_size // 2, x.shape[-1] - (self.sliding_window_size // 2))
        x = np.take(x, indx, axis=-2)
        x = np.take(x, indy, axis=-1)
        # reshape back to 1d
        x = self.reshape_input_2d_inverse(x, n_tiles=n_tiles)
        return x

    def aggregate_conv2d(self, x, n_tiles=1, size=None, padding="valid", mean=False):
        # conv2d expects 4d input: [batch, channels, height, width]
        input = self.reshape_input_2d(x, n_tiles=n_tiles)
        # conv2d expects 4d weights: [out_channels, in_channels/groups, height, width]
        if size is None:
            size = self.sliding_window_size
        weights = torch.ones((x.shape[-1], 1, size, size), device=input.device)
        if mean:
            weights = weights / torch.tensor(size * size, device=input.device)
        x = torch.nn.functional.conv2d(
            input,
            weights,
            padding=padding,
            groups=x.shape[-1],
        )
        x = self.reshape_input_2d_inverse(x, n_tiles=n_tiles)
        return x

    def learnable_neighbour_effect_conv2d(self, x, name, n_tiles=1, size=None, n_out=None, padding="valid"):
        # pyro version

        # conv2d expects 4d input: [batch, channels, height, width]
        input = self.reshape_input_2d(x, n_tiles=n_tiles)
        # conv2d expects 4d weights: [out_channels, in_channels/groups, height, width]
        if n_out is None:
            n_out = x.shape[-1]
            groups = x.shape[-1]
        else:
            groups = 1
        if size is None:
            size = self.sliding_window_size
        weights_shape = [n_out, int(x.shape[-1] / groups), size, size]
        weights = pyro.sample(
            f"{name}_weights",
            dist.SoftLaplace(self.zeros, self.ones).expand(weights_shape).to_event(len(weights_shape)),
        )  # [self.n_factors, self.n_factors]
        x = torch.nn.functional.conv2d(
            input,
            weights,
            padding=padding,
            groups=groups,
        )
        x = self.reshape_input_2d_inverse(x, n_tiles=n_tiles)
        return x

    def redistribute_conv2d(self, x, name, n_tiles=1, size=None, n_out=None, padding="same"):
        # pyro version

        # conv2d expects 4d input: [batch, channels, height, width]
        input = self.reshape_input_2d(x, n_tiles=n_tiles)
        # conv2d expects 4d weights: [out_channels, in_channels/groups, height, width]
        if n_out is None:
            n_out = x.shape[-1]
            groups = x.shape[-1]
        else:
            groups = 1
        if size is None:
            size = self.sliding_window_size
        weights_shape = [n_out, int(n_out / groups)]
        weights = pyro.sample(
            f"{name}_weights",
            dist.Dirichlet(self.ones_1d.expand([size * size]))
            .expand(weights_shape)
            .to_event(reinterpreted_batch_ndims=None),
        )  # [self.n_factors, self.n_factors]
        weights = rearrange(weights, "o g (s z) -> o g s z", s=size, z=size)
        x = torch.nn.functional.conv2d(
            input,
            weights,
            padding=padding,
            groups=groups,
        )
        x = self.reshape_input_2d_inverse(x, n_tiles=n_tiles)
        return x

    def learnable_neighbour_effect_conv2d_nn(
        self,
        x,
        name,
        n_tiles=1,
        size=None,
        n_out=None,
        padding="valid",
        use_weigted_cnn_weights=None,
    ):
        # pure pytorch version

        if use_weigted_cnn_weights is None:
            use_weigted_cnn_weights = self.use_weigted_cnn_weights

        # dropout
        x = self.dropout(x)

        # conv2d expects 4d input: [batch, channels, height, width]
        input = self.reshape_input_2d(x, n_tiles=n_tiles)
        # conv2d expects 4d weights: [out_channels, in_channels/groups, height, width]
        if n_out is None:
            n_out = x.shape[-1]
            groups = x.shape[-1]
        else:
            groups = 1
        if size is None:
            size = self.sliding_window_size
        if False:
            if getattr(self.weights, name + "_g", None) is None:
                # gene-wise weights
                init_param = torch.normal(
                    torch.full(
                        size=[1, x.shape[-1], 1, 1],
                        fill_value=0.0,
                        device=input.device,
                    ),
                    torch.full(
                        size=[1, x.shape[-1], 1, 1],
                        fill_value=1.0 / np.sqrt(self.cell_state.shape[1]),
                        device=input.device,
                    ),
                )
                deep_setattr(
                    self.weights,
                    name + "_g",
                    pyro.nn.PyroParam(
                        init_param.to(input.device).requires_grad_(True),
                        constraint=torch.distributions.constraints.positive,
                    ),
                )
            weights_g = deep_getattr(self.weights, name + "_g").to(input.device)
            input = input * weights_g
        if use_weigted_cnn_weights and (groups == 1):
            weights = self.cell_state / (
                self.cell_state.sum(0, keepdims=True) + torch.tensor(1e-4, device=input.device)
            )
            n_repeats = np.ceil(n_out / self.cell_state.shape[0])
            weights = torch.tile(weights, (int(n_repeats), 1))[:n_out, :]
            weights = torch.tile(weights.unsqueeze(-1).unsqueeze(-1), (1, 1, size, size)).detach()
            if getattr(self.weights, name, None) is None:
                init_param = torch.normal(
                    torch.full(
                        size=(n_out, self.cell_state.shape[1], size, size),
                        fill_value=0.0,
                        device=input.device,
                    ),
                    torch.full(
                        size=(n_out, self.cell_state.shape[1], size, size),
                        fill_value=1.0 / np.sqrt(size * size * self.cell_state.shape[1]),
                        device=input.device,
                    ),
                )
                deep_setattr(
                    self.weights,
                    name,
                    pyro.nn.PyroParam(
                        init_param,  # .to(input.device).requires_grad_(True),
                    ),
                )
            weights_ = deep_getattr(self.weights, name).to(input.device)
            x = torch.nn.functional.conv2d(
                input,
                weights * weights_,
                padding=padding,
                groups=groups,
            )
        else:
            if getattr(self.weights, name, None) is None:
                deep_setattr(
                    self.weights,
                    name,
                    PyroModule[torch.nn.Conv2d](
                        in_channels=x.shape[-1],
                        out_channels=n_out,
                        kernel_size=size,
                        padding=padding,
                        groups=groups,
                    ).to(input.device),
                )
            mod = deep_getattr(self.weights, name).to(input.device)
            x = mod(input)
        x = self.reshape_input_2d_inverse(x, n_tiles=n_tiles)
        if getattr(self.weights, name + "_layer_norm", None) is None:
            deep_setattr(
                self.weights,
                name + "_layer_norm",
                PyroModule[torch.nn.LayerNorm](x.shape[-1], elementwise_affine=False).to(input.device),
            )
        mod = deep_getattr(self.weights, name + "_layer_norm").to(input.device)
        x = mod(x)
        x = torch.nn.functional.softplus(x)
        return x

    def learnable_neighbour_effect_conv2d_nn_layers(self, x, name, n_tiles=1, size=None, n_out=None, padding="valid"):
        # pure pytorch version
        # n_layers = 2
        x = self.learnable_neighbour_effect_conv2d_nn(
            x=x, name=name, n_tiles=n_tiles, size=size, n_out=n_out, padding=padding
        )
        for i in [2]:
            x = x + self.learnable_neighbour_effect_conv2d_nn(
                x=x,
                name=f"{name}_{i}",
                n_tiles=n_tiles,
                size=size,
                n_out=n_out,
                padding=padding,
                use_weigted_cnn_weights=False,
            )

        return x

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
                    n_signals=len(self.signal_bool),
                    n_receptors=len(self.receptor_bool),
                    n_out=n_out,
                    n_pathways=self.n_pathways,
                    signal_receptor_mask=self.signal_receptor_mask,  # tells which receptors can bind which ligands
                    receptor_tf_mask=self.receptor_tf_mask,  # tells which receptors can influence which TF (eg nuclear receptor = TF)
                    dropout_rate=self.dropout_rate,
                    use_horseshoe_prior=True,
                    use_gamma_horseshoe_prior=False,
                    weights_prior_tau=0.1,
                    use_pathway_interaction_effect=self.use_pathway_interaction_effect,
                    average_distance_prior=average_distance_prior,
                    use_non_negative_weights=self.use_non_negative_weights,
                ),
            )
        # get module
        return deep_getattr(self.weights, name)

    def get_lr_abundance(self, d_sg, m_g, y_s):
        # get lr abundance
        signal_abundance = d_sg[:, self.signal_bool] / y_s
        receptor_abundance = torch.minimum(
            (self.cell_state * m_g)[:, self.receptor_bool], (self.cell_state * m_g)[:, self.receptor_bool_b]
        )

        return signal_abundance.detach(), receptor_abundance.detach()

    def cell_comm_effect(
        self,
        signal_abundance,
        receptor_abundance,
        distances,
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
            obs_plate,
        )
        # compute cell abundance prediction
        w_sf_mu = module.signal_receptor_tf_effect_spatial(
            bound_receptor_abundance_src,
        )
        return w_sf_mu

    def n_cells_per_location_prior(self, obs_plate):
        if self.N_cells_per_location.shape[0] == self.n_obs:
            with obs_plate as ind:
                N_cells_per_location = self.N_cells_per_location[ind, :]
            if self.N_cells_per_location_alpha_prior is not None:
                n_s_cells_per_location_prior = self.N_cells_per_location_alpha_prior
        else:
            N_cells_per_location = self.N_cells_per_location
            if self.N_cells_per_location_alpha_prior is not None:
                n_s_cells_per_location_prior = pyro.sample(  # 1/2
                    "n_s_cells_per_location_prior",
                    dist.Exponential(
                        self.N_cells_per_location_alpha_prior * self.ones,  # 2
                    )
                    .expand([1, 1])
                    .to_event(2),
                )
                n_s_cells_per_location_prior = self.ones / n_s_cells_per_location_prior.pow(2)  # 4
        if self.N_cells_per_location_alpha_prior is not None:
            with obs_plate:
                n_s_cells_per_location = pyro.sample(
                    "n_s_cells_per_location",
                    dist.Gamma(
                        n_s_cells_per_location_prior,
                        n_s_cells_per_location_prior / N_cells_per_location,
                    ),
                )
        else:
            with obs_plate:
                # prior on number of cells per location
                n_s_cells_per_location = pyro.sample(
                    "n_s_cells_per_location",
                    dist.Gamma(
                        N_cells_per_location * self.N_cells_mean_var_ratio,
                        self.N_cells_mean_var_ratio,
                    ),
                )
        return n_s_cells_per_location

    def a_s_factors_per_location_prior(self, obs_plate):
        if self.A_B_per_location_alpha_prior is not None:
            a_s_factors_per_location_prior = pyro.sample(  # 1/2
                "a_s_factors_per_location_prior",
                dist.Exponential(
                    self.A_B_per_location_alpha_prior * self.ones,  # 2
                )
                .expand([1, 1])
                .to_event(2),
            )
            a_s_factors_per_location_prior = self.ones / a_s_factors_per_location_prior.pow(2)  # 4
            with obs_plate:
                a_s_factors_per_location = pyro.sample(
                    "a_s_factors_per_location",
                    dist.Gamma(
                        a_s_factors_per_location_prior,
                        a_s_factors_per_location_prior / self.A_factors_per_location,
                    ),
                )
        else:
            with obs_plate:
                # prior on number of cells per location
                a_s_factors_per_location = pyro.sample(
                    "a_s_factors_per_location",
                    dist.Gamma(
                        self.A_factors_per_location * self.ones,
                        self.ones,
                    ),
                )
        return a_s_factors_per_location

    def b_s_groups_per_location_prior(self, obs_plate):
        if self.A_B_per_location_alpha_prior is not None:
            b_s_groups_per_location_prior = pyro.sample(  # 1/2
                "b_s_groups_per_location_prior",
                dist.Exponential(
                    self.A_B_per_location_alpha_prior * self.ones,  # 2
                )
                .expand([1, 1])
                .to_event(2),
            )
            b_s_groups_per_location_prior = self.ones / b_s_groups_per_location_prior.pow(2)  # 4
            with obs_plate:
                b_s_groups_per_location = pyro.sample(
                    "b_s_groups_per_location",
                    dist.Gamma(
                        b_s_groups_per_location_prior,
                        b_s_groups_per_location_prior / self.B_groups_per_location,
                    ),
                )
        else:
            with obs_plate:
                # prior on number of cells per location
                b_s_groups_per_location = pyro.sample(
                    "b_s_groups_per_location",
                    dist.Gamma(
                        self.B_groups_per_location * self.ones,
                        self.ones,
                    ),
                )
        return b_s_groups_per_location

    def factorisation_prior_on_w_sf(self, obs_plate):
        # factorisation prior on w_sf models similarity in locations
        # across cell types f and reflects the absolute scale of w_sf
        n_s_cells_per_location = self.n_cells_per_location_prior(obs_plate)
        b_s_groups_per_location = self.b_s_groups_per_location_prior(obs_plate)

        # cell group loadings
        shape = self.ones_1_n_groups * b_s_groups_per_location / self.n_groups_tensor
        rate = self.ones_1_n_groups / (n_s_cells_per_location / b_s_groups_per_location)
        with obs_plate:
            k = "z_sr_groups_factors"
            z_sr_groups_factors = pyro.sample(
                k,
                dist.Gamma(shape, rate),  # .to_event(1)#.expand([self.n_groups]).to_event(1)
            )  # (n_obs, n_groups)

        k_r_factors_per_groups = pyro.sample(
            "k_r_factors_per_groups",
            dist.Gamma(self.factors_per_groups, self.ones).expand([self.n_groups, 1]).to_event(2),
        )  # (self.n_groups, 1)

        c2f_shape = k_r_factors_per_groups / self.n_factors_tensor

        x_fr_group2fact = pyro.sample(
            "x_fr_group2fact",
            dist.Gamma(c2f_shape, k_r_factors_per_groups).expand([self.n_groups, self.n_factors]).to_event(2),
        )  # (self.n_groups, self.n_factors)

        w_sf_mu = z_sr_groups_factors @ x_fr_group2fact

        return w_sf_mu

    def proportion_factorisation_prior_on_w_sf_v1(self, obs_plate):
        # factorisation prior on w_sf models similarity in locations
        # across cell types f and reflects the absolute scale of w_sf
        n_s_cells_per_location = self.n_cells_per_location_prior(obs_plate)
        b_s_groups_per_location = self.b_s_groups_per_location_prior(obs_plate)

        # cell group loadings
        shape = self.ones_1_n_groups * b_s_groups_per_location / self.n_groups_tensor
        rate = self.ones_1_n_groups
        with obs_plate:
            k = "z_sr_groups_factors"
            z_sr_groups_factors = pyro.sample(
                k,
                dist.Gamma(shape, rate),  # .to_event(1)#.expand([self.n_groups]).to_event(1)
            )  # (n_obs, n_groups)

        c2f_shape = self.factors_per_groups / self.n_factors_tensor

        x_fr_lambdas_group2fact = pyro.sample(
            "x_fr_lambdas_group2fact",
            dist.Gamma(c2f_shape, self.ones_1_n_groups.T).expand([self.n_groups, self.n_factors]).to_event(2),
        )  # (self.n_groups, self.n_factors)
        x_fr_weights_group2fact = pyro.sample(
            "x_fr_weights_group2fact",
            dist.Normal(self.zeros, self.ones_1_n_groups.T).expand([self.n_groups, self.n_factors]).to_event(2),
        )  # (self.n_groups, self.n_factors)

        w_sf_mu = z_sr_groups_factors @ (x_fr_lambdas_group2fact * x_fr_weights_group2fact)
        w_sf_mu = w_sf_mu * torch.tensor(100.0, device=w_sf_mu.device)
        # print("w_sf_mu 1 mean", w_sf_mu.mean().item(), "w_sf_mu 1 std", w_sf_mu.std().item())
        # print("w_sf_mu 1 min", w_sf_mu.min().item(), "w_sf_mu 1 max", w_sf_mu.max().item())

        w_sf_mu = torch.softmax(w_sf_mu, dim=-1)
        # print("w_sf_mu sum dim -2", w_sf_mu.sum(dim=-2).shape, w_sf_mu.sum(dim=-2))
        # print("w_sf_mu sum dim -1", w_sf_mu.sum(dim=-1).shape, w_sf_mu.sum(dim=-1))
        # print("w_sf_mu mean", w_sf_mu.mean().item(), "w_sf_mu std", w_sf_mu.std().item())
        # print("w_sf_mu min", w_sf_mu.min().item(), "w_sf_mu max", w_sf_mu.max().item())
        w_sf_mu = w_sf_mu * n_s_cells_per_location

        return w_sf_mu

    def proportion_factorisation_prior_on_w_sf(self, obs_plate):
        # factorisation prior on w_sf models similarity in locations
        # across cell types f and reflects the absolute scale of w_sf
        n_s_cells_per_location = self.n_cells_per_location_prior(obs_plate)
        b_s_groups_per_location = self.b_s_groups_per_location_prior(obs_plate)

        # cell group loadings
        shape = self.ones_1_n_groups * b_s_groups_per_location / self.n_groups_tensor
        rate = self.ones_1_n_groups
        with obs_plate:
            k = "z_sr_groups_factors"
            z_sr_groups_factors = pyro.sample(
                k,
                dist.Gamma(shape, rate),  # .to_event(1)#.expand([self.n_groups]).to_event(1)
            )  # (n_obs, n_groups)

        c2f_shape = self.factors_per_groups / self.n_factors_tensor

        x_fr_lambdas_group2fact = pyro.sample(
            "x_fr_lambdas_group2fact",
            dist.Gamma(c2f_shape, self.ones_1_n_groups.T).expand([self.n_groups, self.n_factors]).to_event(2),
        )  # (self.n_groups, self.n_factors)

        w_sf_mu = z_sr_groups_factors @ x_fr_lambdas_group2fact
        w_sf_mu = w_sf_mu * torch.tensor(100.0, device=w_sf_mu.device)
        # print("w_sf_mu 1 mean", w_sf_mu.mean().item(), "w_sf_mu 1 std", w_sf_mu.std().item())
        # print("w_sf_mu 1 min", w_sf_mu.min().item(), "w_sf_mu 1 max", w_sf_mu.max().item())

        w_sf_mu = w_sf_mu / w_sf_mu.sum(dim=-1, keepdim=True)
        # print("w_sf_mu sum dim -2", w_sf_mu.sum(dim=-2).shape, w_sf_mu.sum(dim=-2))
        # print("w_sf_mu sum dim -1", w_sf_mu.sum(dim=-1).shape, w_sf_mu.sum(dim=-1))
        # print("w_sf_mu mean", w_sf_mu.mean().item(), "w_sf_mu std", w_sf_mu.std().item())
        # print("w_sf_mu min", w_sf_mu.min().item(), "w_sf_mu max", w_sf_mu.max().item())
        w_sf_mu = w_sf_mu * n_s_cells_per_location

        return w_sf_mu

    def independent_prior_on_w_sf(self, obs_plate):
        n_s_cells_per_location = self.n_cells_per_location_prior(obs_plate)
        a_s_factors_per_location = self.a_s_factors_per_location_prior(obs_plate)

        # cell group loadings
        shape = self.ones_1_n_factors * a_s_factors_per_location / self.n_factors_tensor
        rate = self.ones_1_n_factors / (n_s_cells_per_location / a_s_factors_per_location)

        with obs_plate:
            w_sf = pyro.sample(
                "prior_w_sf",
                dist.Gamma(
                    shape,
                    rate,
                ),
            )  # (self.n_obs, self.n_factors)
        return w_sf

    def forward(
        self,
        x_data,
        idx,
        batch_index,
        tiles: torch.Tensor = None,
        tiles_unexpanded: torch.Tensor = None,
        positions: torch.Tensor = None,
        in_tissue: torch.Tensor = None,
        normalising_factor_y_s: Optional[torch.Tensor] = None,
        x_data_normalised: Optional[torch.Tensor] = None,
    ):
        if tiles_unexpanded is not None:
            tiles_in_use = tiles.sum(0).bool()
            obs_in_use = tiles_unexpanded[:, tiles_in_use].sum(1).bool()
            idx = idx[obs_in_use]
            batch_index = batch_index[obs_in_use]
            if positions is not None:
                positions = positions[obs_in_use]
        # if self.sliding_window_size > 0:
        #    # remove observations that will not be included after convolution with padding='valid'
        #    idx = self.crop_according_to_valid_padding(idx.unsqueeze(-1)).squeeze(-1)
        #    batch_index = self.crop_according_to_valid_padding(batch_index)
        #    if positions is not None:
        #        positions = self.crop_according_to_valid_padding(positions)
        obs2sample = one_hot(batch_index, self.n_batch).float()
        obs_plate = self.create_plates(
            x_data,
            idx,
            batch_index,
            tiles,
            tiles_unexpanded,
            positions,
            in_tissue,
            normalising_factor_y_s,
            x_data_normalised,
        )
        if tiles is not None:
            n_tiles = tiles.shape[1]
        else:
            n_tiles = 1
        if in_tissue is None:
            in_tissue = self.ones_1d.expand((x_data.shape[0], 1)).bool()

        if getattr(self, "distances", None) is not None:
            distances = self.distances
        elif positions is not None:
            # compute distance using positions [observations, 2]
            distances = (
                (positions.unsqueeze(1) - positions.unsqueeze(0))  # [observations, 1, 2]  # [1, observations, 2]
                .pow(2)
                .sum(-1)
                .sqrt()
            ) + torch.tensor(25.0, device=positions.device)

        # =====================Gene expression level scaling m_g======================= #
        # Explains difference in sensitivity for each gene between single cell and spatial technology
        m_g_mean = pyro.sample(
            "m_g_mean",
            dist.Gamma(
                self.m_g_mu_mean_var_ratio_hyp * self.m_g_mu_hyp,
                self.m_g_mu_mean_var_ratio_hyp,
            )
            .expand([1, 1])
            .to_event(2),
        )  # (1, 1)

        m_g_alpha_e_inv = pyro.sample(
            "m_g_alpha_e_inv",
            dist.Exponential(self.m_g_alpha_hyp_mean).expand([1, 1]).to_event(2),
        )  # (1, 1)
        m_g_alpha_e = self.ones / m_g_alpha_e_inv.pow(2)

        m_g = pyro.sample(
            "m_g",
            dist.Gamma(m_g_alpha_e, m_g_alpha_e / m_g_mean).expand([1, self.n_vars]).to_event(2),  # self.m_g_mu_hyp)
        )  # (1, n_vars)

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

        if normalising_factor_y_s is not None:
            detection_y_s = detection_y_s * normalising_factor_y_s
            pyro.deterministic("total_detection_y_s", detection_y_s)

        # if (self.sliding_window_size > 0) and self.use_aggregated_detection_y_s:
        #    detection_y_s = self.aggregate_conv2d(
        #        detection_y_s,
        #        padding="same",
        #        mean=True,
        #        size=20,
        #        n_tiles=n_tiles,
        #    )
        #    pyro.deterministic("aggregated_detection_y_s", detection_y_s)

        # =====================Cell abundances w_sf======================= #
        if not (self.use_cell_comm_prior_on_w_sf or self.use_independent_prior_on_w_sf):
            if self.use_proportion_factorisation_prior_on_w_sf:
                w_sf_mu = self.proportion_factorisation_prior_on_w_sf(obs_plate)
            else:
                w_sf_mu = self.factorisation_prior_on_w_sf(obs_plate)
            if self.use_learnable_mean_var_ratio:
                w_sf_mean_var_ratio_hyp = pyro.sample(
                    "w_sf_mean_var_ratio_hyp",
                    dist.Gamma(self.w_sf_mean_var_ratio_tensor, self.ones).expand([1, 1]).to_event(2),
                )
                w_sf_mean_var_ratio = pyro.sample(
                    "w_sf_mean_var_ratio",
                    dist.Exponential(w_sf_mean_var_ratio_hyp).expand([1, self.n_factors]).to_event(2),
                )  # (self.n_batch, self.n_vars)
                w_sf_mean_var_ratio = self.ones / (
                    w_sf_mean_var_ratio + torch.tensor(1.0 / 20.0, device=w_sf_mean_var_ratio.device)
                )
            else:
                w_sf_mean_var_ratio = self.w_sf_mean_var_ratio_tensor
            with obs_plate:
                k = "w_sf"
                w_sf = pyro.sample(
                    k,
                    dist.Gamma(
                        w_sf_mu * w_sf_mean_var_ratio,
                        w_sf_mean_var_ratio,
                    ),
                )  # (self.n_obs, self.n_factors)
        elif self.use_cell_comm_prior_on_w_sf:
            w_sf_mu = self.independent_prior_on_w_sf(obs_plate)
            # get lr abundance
            signal_abundance, receptor_abundance = self.get_lr_abundance(
                x_data,
                m_g,
                detection_y_s,
            )
            w_sf_mu_cell_comm = self.cell_comm_effect(
                signal_abundance=signal_abundance,
                receptor_abundance=receptor_abundance,
                distances=distances,
                average_distance_prior=self.average_distance_prior,
                obs_plate=obs_plate,
            )
            w_sf_mean_var_ratio_hyp = pyro.sample(
                "w_sf_mean_var_ratio_hyp",
                dist.Gamma(self.w_sf_mean_var_ratio_tensor, self.ones).expand([1, 1]).to_event(2),
            )
            w_sf_mean_var_ratio = pyro.sample(
                "w_sf_mean_var_ratio",
                dist.Exponential(w_sf_mean_var_ratio_hyp).expand([1, self.n_factors]).to_event(2),
            )  # (1, self.n_vars)
            w_sf_mean_var_ratio = self.ones / (
                w_sf_mean_var_ratio + torch.tensor(1.0 / 20.0, device=w_sf_mean_var_ratio.device)
            )
            if tiles_unexpanded is not None:
                x_data = x_data[obs_in_use]
                w_sf_mu_cell_comm = w_sf_mu_cell_comm[obs_in_use]
            with obs_plate:
                k = "w_sf"
                pyro.deterministic(k, w_sf_mu)  # (self.n_obs, self.n_factors)
                pyro.deterministic(f"{k}_cell_comm", w_sf_mu_cell_comm)  # (self.n_obs, self.n_factors)
                w_sf = pyro.sample(
                    f"{k}_obs",
                    dist.Gamma(
                        w_sf_mu_cell_comm * w_sf_mean_var_ratio,
                        w_sf_mean_var_ratio,
                    ),
                    obs=w_sf_mu,
                )  # (self.n_obs, self.n_factors)
        elif self.use_independent_prior_on_w_sf:
            w_sf_mu = self.independent_prior_on_w_sf(obs_plate)
            with obs_plate:
                k = "w_sf"
                w_sf = pyro.deterministic(k, w_sf_mu)  # (self.n_obs, self.n_factors)

        if self.use_cell_comm_likelihood_w_sf:
            signal_abundance, receptor_abundance = self.get_lr_abundance(
                x_data,
                m_g,
                detection_y_s,
            )
            w_sf_mu_cell_comm = self.cell_comm_effect(
                signal_abundance=signal_abundance,
                receptor_abundance=receptor_abundance,
                distances=distances,
                average_distance_prior=self.average_distance_prior,
                obs_plate=obs_plate,
            )
            w_sf_mean_var_ratio_hyp = pyro.sample(
                "w_sf_mean_var_ratio_hyp_lik",
                dist.Gamma(self.w_sf_mean_var_ratio_tensor, self.ones).expand([1, 1]).to_event(2),
            )
            w_sf_mean_var_ratio = pyro.sample(
                "w_sf_mean_var_ratio_lik",
                dist.Exponential(w_sf_mean_var_ratio_hyp).expand([1, self.n_factors]).to_event(2),
            )  # (self.n_batch, self.n_vars)
            w_sf_mean_var_ratio = self.ones / (
                w_sf_mean_var_ratio + torch.tensor(1.0 / 20.0, device=w_sf_mean_var_ratio.device)
            )
            if tiles_unexpanded is not None:
                x_data = x_data[obs_in_use]
                w_sf_mu_cell_comm = w_sf_mu_cell_comm[obs_in_use]
            with obs_plate:
                k = "w_sf"
                pyro.deterministic(f"{k}_cell_comm", w_sf_mu_cell_comm)  # (self.n_obs, self.n_factors)
                pyro.sample(
                    f"{k}_obs",
                    dist.Gamma(
                        w_sf_mu_cell_comm * w_sf_mean_var_ratio,
                        w_sf_mean_var_ratio,
                    ),
                    obs=w_sf.detach(),
                )  # (self.n_obs, self.n_factors)

        if self.use_cell_compartments:
            with obs_plate:
                k = "cell_compartment_w_sfk"
                w_sfk = pyro.sample(
                    k,
                    dist.Dirichlet(self.ones_1d.expand((self.n_factors, self.n_cell_compartments))),
                )  # ( self.n_factors, self.n_obs, self.n_cell_compartments)
            w_sf = torch.einsum("sfk,sf->fsk", w_sfk, w_sf)

            if (self.sliding_window_size > 0) and self.use_aggregated_w_sf:
                w_sf = self.redistribute_conv2d(
                    rearrange(w_sf, "f s k -> s (f k)"),
                    n_tiles=n_tiles,
                    name="redistribute",
                    padding="same",
                )
                w_sf = rearrange(w_sf, "s (f k) -> f s k", f=self.n_factors, k=self.n_cell_compartments)
                pyro.deterministic("aggregated_w_fsk", w_sf)
            if self.use_per_cell_type_normalisation:
                per_cell_type_normalisation_f = pyro.sample(
                    "per_cell_type_normalisation_f",
                    dist.Gamma(self.detection_cell_type_prior_alpha, self.detection_cell_type_prior_alpha)
                    .expand([self.n_factors, 1, self.n_cell_compartments])
                    .to_event(3),
                )  # [self.n_factors, 1, self.n_cell_compartments]
                w_sf = w_sf * per_cell_type_normalisation_f
        else:
            if (self.sliding_window_size > 0) and self.use_aggregated_w_sf:
                w_sf = self.redistribute_conv2d(
                    w_sf,
                    name="redistribute",
                    padding="same",
                    n_tiles=n_tiles,
                )
                pyro.deterministic("aggregated_w_sf", w_sf)
            if self.use_per_cell_type_normalisation:
                per_cell_type_normalisation_f = pyro.sample(
                    "per_cell_type_normalisation_f",
                    dist.Gamma(self.detection_cell_type_prior_alpha, self.detection_cell_type_prior_alpha)
                    .expand([1, self.n_factors])
                    .to_event(2),
                )  # (1, self.n_factors)
                w_sf = w_sf * per_cell_type_normalisation_f

        # =====================Gene-specific additive component ======================= #
        # per gene molecule contribution that cannot be explained by
        # cell state signatures (e.g. background, free-floating RNA)
        s_g_gene_add_alpha_hyp = pyro.sample(
            "s_g_gene_add_alpha_hyp",
            dist.Gamma(self.ones * self.gene_add_alpha_hyp_prior_alpha, self.ones * self.gene_add_alpha_hyp_prior_beta)
            .expand([1, 1])
            .to_event(2),
        )
        s_g_gene_add_mean = pyro.sample(
            "s_g_gene_add_mean",
            dist.Gamma(
                self.gene_add_mean_hyp_prior_alpha,
                self.gene_add_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )  # (self.n_batch)
        s_g_gene_add_alpha_e_inv = pyro.sample(
            "s_g_gene_add_alpha_e_inv",
            dist.Exponential(s_g_gene_add_alpha_hyp).expand([self.n_batch, 1]).to_event(2),
        )  # (self.n_batch)
        s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)

        s_g_gene_add = pyro.sample(
            "s_g_gene_add",
            dist.Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean)
            .expand([self.n_batch, self.n_vars])
            .to_event(2),
        )  # (self.n_batch, n_vars)

        # =====================Gene-specific overdispersion ======================= #
        alpha_g_phi_hyp = pyro.sample(
            "alpha_g_phi_hyp",
            dist.Gamma(self.ones * self.alpha_g_phi_hyp_prior_alpha, self.ones * self.alpha_g_phi_hyp_prior_beta)
            .expand([1, 1])
            .to_event(2),
        )
        alpha_g_inverse = pyro.sample(
            "alpha_g_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([self.n_batch, self.n_vars]).to_event(2),
        )  # (self.n_batch, self.n_vars)

        # =====================Expected expression ======================= #
        if not self.training_wo_observed:
            # expected expression
            if self.use_cell_compartments:
                k = "cell_compartment_g_fgk"
                cell_compartment_g_fgk = pyro.sample(
                    k,
                    dist.Dirichlet(self.ones_1d.expand((1, 1, self.n_cell_compartments)))
                    .expand([self.n_factors, self.n_vars])
                    .to_event(reinterpreted_batch_ndims=None),
                )  # ( self.n_factors, self.n_vars, self.n_cell_compartments)
                mu = torch.einsum("fsk,fgk,fg->sg", w_sf, cell_compartment_g_fgk, self.cell_state)
            else:
                mu = w_sf @ self.cell_state
            mu = (mu * m_g + (obs2sample @ s_g_gene_add)) * detection_y_s
            alpha = obs2sample @ (self.ones / alpha_g_inverse.pow(2))
            # convert mean and overdispersion to total count and logits
            # total_count, logits = _convert_mean_disp_to_counts_logits(
            #    mu, alpha, eps=self.eps
            # )

            # =====================DATA likelihood ======================= #
            # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
            # if self.dropout_p != 0:
            #    x_data = self.dropout(x_data)
            if not self.sliding_window_size_list_exist:
                if self.sliding_window_size > 0:
                    x_data = self.aggregate_conv2d(
                        x_data,
                        padding="same",
                        n_tiles=n_tiles,
                        size=self.sliding_window_size,
                    )
                with obs_plate:
                    pyro.sample(
                        "data_target",
                        dist.GammaPoisson(concentration=alpha, rate=alpha / mu),
                        obs=x_data,
                    )
            else:
                for i, size in enumerate(self.sliding_window_size_list):
                    if self.sliding_window_size_list[i] > 0:
                        mu_ = self.aggregate_conv2d(
                            mu,
                            padding="same",
                            n_tiles=n_tiles,
                            size=size,
                        )
                        alpha_ = alpha * torch.tensor((self.sliding_window_size_list[i] ** 2) / 100, device=mu.device)
                        # alpha_g_size_effect = pyro.sample(
                        #    f"alpha_g_size_{size}",
                        #    dist.Gamma(self.ones + self.ones, self.ones + self.ones).to_event(2),
                        # )
                        # alpha_ = alpha_ * alpha_g_size_effect
                        x_data_ = self.aggregate_conv2d(
                            x_data,
                            padding="same",
                            n_tiles=n_tiles,
                            size=size,
                        )
                    else:
                        mu_ = mu
                        alpha_ = alpha * torch.tensor((1**2) / 100, device=mu.device)
                        x_data_ = x_data
                    with obs_plate, pyro.poutine.mask(mask=in_tissue):
                        pyro.sample(
                            f"data_target_{size}",
                            dist.GammaPoisson(concentration=alpha_, rate=alpha_ / mu_),
                            obs=x_data_,
                        )

        # =====================Compute mRNA count from each factor in locations  ======================= #
        with obs_plate:
            if not self.training:
                if self.use_cell_compartments:
                    mRNA = torch.einsum("fsk,fgk,fg->fsk", w_sf, cell_compartment_g_fgk, self.cell_state * m_g)
                    pyro.deterministic("u_fsk_mRNA_factors", mRNA)
                else:
                    mRNA = w_sf * (self.cell_state * m_g).sum(-1)
                    pyro.deterministic("u_sf_mRNA_factors", mRNA)

    def compute_expected(
        self,
        samples,
        adata_manager,
        ind_x=None,
        hide_ambient=False,
        hide_cell_type=False,
    ):
        r"""Compute expected expression of each gene in each location. Useful for evaluating how well
        the model learned expression pattern of all genes in the data.
        """
        if ind_x is None:
            ind_x = np.arange(adata_manager.adata.n_obs).astype(int)
        else:
            ind_x = ind_x.astype(int)
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten()).values[ind_x, :]
        mu = np.ones((1, 1))
        if not hide_cell_type:
            mu = np.dot(samples["w_sf"][ind_x, :], self.cell_state_mat.T) * samples["m_g"]
        if not hide_ambient:
            mu = mu + np.dot(obs2sample, samples["s_g_gene_add"])
        mu = mu * samples["detection_y_s"][ind_x, :]
        alpha = np.dot(obs2sample, 1 / np.power(samples["alpha_g_inverse"], 2))

        return {"mu": mu, "alpha": alpha, "ind_x": ind_x}

    def compute_expected_per_cell_type(self, samples, adata_manager, ind_x=None):
        r"""
        Compute expected expression of each gene in each location for each cell type.

        Parameters
        ----------
        samples
            Posterior distribution summary self.samples[f"post_sample_q05}"]
            (or 'means', 'stds', 'q05', 'q95') produced by export_posterior().
        ind_x
            Location/observation indices for which to compute expected count
            (if None all locations are used).

        Returns
        -------
        dict
          dictionary with:

            1. list with expected expression counts (sparse, shape=(N locations, N genes)
               for each cell type in the same order as mod\.factor_names_;
            2. np.array with location indices
        """
        if ind_x is None:
            ind_x = np.arange(adata_manager.adata.n_obs).astype(int)
        else:
            ind_x = ind_x.astype(int)

        # fetch data
        x_data = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)[ind_x, :]
        x_data = csr_matrix(x_data)

        # compute total expected expression
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten()).values[ind_x, :]
        mu = np.dot(samples["w_sf"][ind_x, :], self.cell_state_mat.T) * samples["m_g"] + np.dot(
            obs2sample, samples["s_g_gene_add"]
        )

        # compute conditional expected expression per cell type
        mu_ct = [
            csr_matrix(
                x_data.multiply(
                    (
                        np.dot(
                            samples["w_sf"][ind_x, i, np.newaxis],
                            self.cell_state_mat.T[np.newaxis, i, :],
                        )
                        * samples["m_g"]
                    )
                    / mu
                )
            )
            for i in range(self.n_factors)
        ]

        return {"mu": mu_ct, "ind_x": ind_x}
