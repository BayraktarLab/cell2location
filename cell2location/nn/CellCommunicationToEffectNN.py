import numpy as np
import pyro
import torch
from einops import rearrange
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr
from pyro.nn import PyroModule
from scvi.nn import one_hot
from torch import nn as nn

from ._mixins import CreateParameterMixin


class CellCommunicationToTfActivityNN(
    PyroModule,
    CreateParameterMixin,
):
    """
    Defining a function that maps signal abundance in the microenvironment
    to TF activity or cell abundance in target cells via receptors expressed by target cells.

    Parameters
    ----------
    n_tfs
        The number of TFs
    n_signals
        The number of signals
    n_receptors
        The number of receptors (both single- and multi-subunit).
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    use_activation
        Whether to have layer activation at last layer or not
    bias
        Whether to learn bias in linear layers or not
    activation_fn
        Which activation function to use
    """

    return_composite_matches = False
    mask_binding_cooperativity = False
    mask_regulatory_cooperativity = False
    tf_features_reparameterised_regularisation = True
    normalise_distance_weights = False  # True False
    use_bayesian_protein_effects = True
    bayesian = False
    use_sqrt_normalisation = True

    promoter_distance_prior = 50
    use_footprinting_masked_tn5_index = -1

    use_cached_effects = False
    cached_effects = dict()

    def __init__(
        self,
        name: str,
        n_tfs: int,
        n_signals: int,
        n_receptors: int,
        n_out: int = 1,
        n_pathways: int = 1,
        mode: str = "signal_receptor_tf_effect",
        signal_receptor_mask: np.ndarray = None,  # tells which receptors can bind which ligands
        receptor_tf_mask: np.ndarray = None,  # tells which receptors can influence which TF (eg nuclear receptor = TF)
        dropout_rate: float = 0.0,
        activation_fn: nn.Module = nn.Softplus,
        weights_prior={"shape": 1.0, "scale": 1.0, "informed_scale": 0.2},
        bias_prior={"mean": 0.0, "sigma": 1.0},
        mode_suffix: str = "_free",
        use_horseshoe_prior: bool = True,
        use_gamma_horseshoe_prior: bool = False,
        weights_prior_tau: float = 1,
        output_transform: str = "proportion",
        use_unbound_concentration: bool = False,
        use_pathway_interaction_effect: bool = True,
        average_distance_prior: float = 50.0,
        use_non_negative_weights: bool = False,
    ):
        super().__init__()

        self.name = name
        self.name_prefix = name
        self.mode = mode
        self.n_tfs = n_tfs
        self.n_signals = n_signals
        self.n_receptors = n_receptors
        self.n_out = n_out
        self.n_pathways = n_pathways
        self.dropout_rate = dropout_rate

        self.activation_fn = activation_fn
        self.weights_prior = weights_prior
        self.bias_prior = bias_prior
        self.mode_suffix = mode_suffix
        self.use_horseshoe_prior = use_horseshoe_prior
        self.use_gamma_horseshoe_prior = use_gamma_horseshoe_prior

        self.output_transform = output_transform
        self.use_unbound_concentration = use_unbound_concentration

        self.use_pathway_interaction_effect = use_pathway_interaction_effect

        self.average_distance_prior = average_distance_prior

        self.use_non_negative_weights = use_non_negative_weights

        self.weights = PyroModule()

        if signal_receptor_mask is None:
            signal_receptor_mask = np.ones((self.n_signals, self.n_receptors))
        assert signal_receptor_mask.shape == (self.n_signals, self.n_receptors), (
            f"signal_receptor_mask shape {signal_receptor_mask.shape} "
            f"does not match n_signals {self.n_signals} and n_receptors {self.n_receptors}"
        )
        from scipy.sparse import coo_matrix

        signal_receptor_mask = coo_matrix(signal_receptor_mask)
        self.signal_receptor_mask_scipy = signal_receptor_mask
        if receptor_tf_mask is not None:
            self.register_buffer(
                "receptor_tf_mask",
                torch.tensor(np.asarray(receptor_tf_mask).astype("float32")),
            )
        else:
            self.receptor_tf_mask = None

        self.register_buffer("n_tfs_tensor", torch.tensor(float(n_tfs)))
        self.register_buffer("n_signals_tensor", torch.tensor(float(n_signals)))
        self.register_buffer("n_receptors_tensor", torch.tensor(float(n_receptors)))
        self.register_buffer("ones", torch.ones(1))
        self.register_buffer("zeros", torch.zeros(1))
        self.register_buffer("ten", torch.tensor(10.0))
        self.register_buffer("weights_prior_shape", torch.tensor(float(self.weights_prior["shape"])))
        self.register_buffer("weights_prior_scale", torch.tensor(float(self.weights_prior["scale"])))
        self.register_buffer(
            "weights_prior_informed_scale",
            torch.tensor(float(self.weights_prior["informed_scale"])),
        )
        self.register_buffer("bias_mean_prior", torch.tensor(float(self.bias_prior["mean"])))
        self.register_buffer("bias_sigma_prior", torch.tensor(float(self.bias_prior["sigma"])))
        self.register_buffer("weights_prior_tau", torch.tensor(float(weights_prior_tau)))

    def get_tf_effect(
        self,
        x,
        name,
        layer,
        weights_shape,
        remove_diagonal=None,
        weights_prior_tau=None,
        use_horseshoe_prior=None,
        non_negative: bool = False,
        upper_triangle: bool = False,
    ):
        if remove_diagonal is None:
            remove_diagonal = self.remove_diagonal

        weights_name = f"{self.name}_{name}_layer_{layer}_protein2effect"

        zero_diag = self.ones.expand(weights_shape)
        if weights_prior_tau is None:
            weights_prior_tau = self.ones if not hasattr(self, "weights_prior_tau") else self.weights_prior_tau
        if use_horseshoe_prior is None:
            use_horseshoe_prior = False if not hasattr(self, "use_horseshoe_prior") else self.use_horseshoe_prior
        weights = self.get_param(
            x=x,
            name=name,
            layer=layer,
            weights_shape=weights_shape,
            bias_shape=[1],
            random_init_scale=1 / np.sqrt(weights_shape[0]),
            bayesian=True,
            use_non_negative_weights=non_negative,
            weights_prior_tau=weights_prior_tau,
            use_horseshoe_prior=use_horseshoe_prior,
        )
        if upper_triangle:
            if len(weights.shape) > 2:
                weights = torch.triu(torch.ones((weights.shape[0], weights[1]))).unsqueeze(-1) * weights
            else:
                weights = torch.triu(weights)

        if len(weights_shape) == 2:
            # [n_in, n_out]
            if remove_diagonal:
                zero_diag = torch.ones(weights_shape[-2], weights_shape[-2], device=weights.device)
                zero_diag = zero_diag.fill_diagonal_(0.0)
        elif len(weights_shape) == 3 and (weights_shape[0] == weights_shape[1]):
            if remove_diagonal:
                zero_diag = torch.ones(weights_shape[-3], weights_shape[-2], device=weights.device)
                zero_diag = zero_diag.fill_diagonal_(0.0)
                zero_diag = zero_diag.unsqueeze(-1)
        if not self.training:
            pyro.deterministic(f"{weights_name}_total_effect", weights * zero_diag)
        return weights * zero_diag

    def get_signal_distance_effect(
        self,
        x,
        layer,
        name,
        weights_shape,
    ):
        # [n_out, n_in]
        weights_name = f"{self.name}_{name}_layer_{layer}_protein2effect"

        sig_distance_effect = self.get_param(
            x=x,
            name=name,
            layer=layer,
            weights_shape=weights_shape,
            bias_shape=[1],
            random_init_scale=1 / np.sqrt(weights_shape[-1]),
            bayesian=True,
            use_non_negative_weights=False,
        )

        if not self.training:
            pyro.deterministic(f"{weights_name}_total_effect", sig_distance_effect)

        return sig_distance_effect

    def get_signal_receptor_effect(
        self,
        x,
        layer,
        weights_shape,
    ):
        # [n_out, n_in]
        name = "signal_receptor_effect"
        weights_name = f"{self.name}_{name}_layer_{layer}_protein2effect"

        if weights_shape is None:
            weights_shape = [len(self.signal_receptor_mask_scipy.data)]

        rec_sig_effect = self.get_param(
            x=x,
            name=name,
            layer=layer,
            weights_shape=weights_shape,
            bias_shape=[1],
            random_init_scale=1 / np.sqrt(self.n_signals),
            bayesian=True,
            weights_prior_shape=torch.tensor(0.2, device=x.device),
            weights_prior_rate=torch.tensor(1.0, device=x.device),
            # sample positive weights
            use_non_negative_weights=True,
        )

        if not self.training:
            pyro.deterministic(f"{weights_name}_total_effect", rec_sig_effect)

        return rec_sig_effect

    def get_signal_receptor_tf_effect(
        self,
        x,
        layer,
        weights_shape,
        name="signal_receptor_tf_effect",
        use_non_negative_weights=None,
    ):
        # [n_tf, n_signals, n_receptors]
        weights_name = f"{self.name}_{name}_layer_{layer}_protein2effect"

        if use_non_negative_weights is None:
            use_non_negative_weights = self.use_non_negative_weights

        tf_sig_rec_tf_effect = self.get_param(
            x=x,
            name=name,
            layer=layer,
            weights_shape=weights_shape,
            bias_shape=[1],
            random_init_scale=1 / np.sqrt(len(self.signal_receptor_mask_scipy.data)),
            bayesian=True,
            # sample positive weights
            use_non_negative_weights=use_non_negative_weights,
        )

        if self.receptor_tf_mask is not None:
            if self.n_out == 1:
                tf_sig_rec_tf_effect = (
                    tf_sig_rec_tf_effect * self.receptor_tf_mask.T[:, self.signal_receptor_mask_scipy.row]
                )
            else:
                tf_sig_rec_tf_effect = torch.einsum(
                    "hrf,rh->hrf",
                    tf_sig_rec_tf_effect,
                    self.receptor_tf_mask[self.signal_receptor_mask_scipy.row, :],
                )

        if not self.training:
            pyro.deterministic(f"{weights_name}_total_effect", tf_sig_rec_tf_effect)

        return tf_sig_rec_tf_effect

    def inverse_sigmoid_lm(self, x, weight, bias):
        # expand shapes correctly
        if x.dim() == 2:
            weight = weight.unsqueeze(-1)
            bias = bias.unsqueeze(-1)
        elif x.dim() == 3:
            weight = weight.unsqueeze(-1).unsqueeze(-1)
            bias = bias.unsqueeze(-1).unsqueeze(-1)
        # compute sigmoid function
        return self.ones - torch.sigmoid(x * weight + bias)

    def gamma_pdf(self, x, concentration, rate, scaling=None):
        # expand shapes correctly
        if x.dim() == 2:
            concentration = concentration.unsqueeze(-1)
            rate = rate.unsqueeze(-1)
            if scaling is not None:
                scaling = scaling.unsqueeze(-1)
        elif x.dim() == 3:
            concentration = concentration.unsqueeze(-1).unsqueeze(-1)
            rate = rate.unsqueeze(-1).unsqueeze(-1)
            if scaling is not None:
                scaling = scaling.unsqueeze(-1).unsqueeze(-1)
        # compute gamma function
        if scaling is None:
            return (
                pyro.distributions.Gamma(
                    concentration=concentration,
                    rate=rate,
                )
                .log_prob(x)
                .exp()
            )
        return (
            pyro.distributions.Gamma(
                concentration=concentration,
                rate=rate,
            )
            .log_prob(x)
            .exp()
            * scaling
        )

    def inverse_sigmoid_distance_function(
        self,
        distances,
        layer,
        weights_shape,
        name,
        mode,
        average_distance_prior=None,
    ):
        if average_distance_prior is None:
            average_distance_prior = self.average_distance_prior

        # sigmoid function =================
        name_ = f"{name}DistanceWeights"  # strictly positive
        weight = self.get_signal_distance_effect(
            x=distances,
            layer=layer,
            name=name_,
            weights_shape=weights_shape,
        )
        # strictly positive
        weight = (
            nn.functional.softplus(weight)
            # prior of ~1/50 (= 1 / (softplus(0) / 35))
            / torch.tensor(0.7, device=distances.device)
        ) / torch.tensor(average_distance_prior, device=distances.device)
        name_ = f"{name}DistanceBias"
        bias = self.get_signal_distance_effect(
            x=distances,
            layer=layer,
            name=name_,
            weights_shape=weights_shape,
        ) - (
            self.ones + self.ones
        )  # prior of -2
        sigmoid_distance_function = self.inverse_sigmoid_lm(distances, weight, bias)

        # gamma function =================
        name_ = f"{name}DistanceGammaConcentration"  # strictly positive
        gamma_concentration = self.get_signal_distance_effect(
            x=distances,
            layer=layer,
            name=name_,
            weights_shape=weights_shape,
        )
        # strictly positive
        gamma_concentration = (
            nn.functional.softplus(gamma_concentration)
            # prior of ~1/50 (= 1 / (softplus(0) / 35))
            / torch.tensor(0.7, device=distances.device)
        )
        name_ = f"{name}DistanceGammaDistance"  # strictly positive
        gamma_distance = self.get_signal_distance_effect(
            x=distances,
            layer=layer,
            name=name_,
            weights_shape=weights_shape,
        )
        # strictly positive
        gamma_distance = (
            nn.functional.softplus(gamma_distance)
            # prior of ~1/50 (= 1 / (softplus(0) / 35))
            / torch.tensor(0.7, device=distances.device)
        )
        gamma_distance = gamma_distance / torch.tensor(average_distance_prior, device=distances.device)
        gamma_distance_function = self.gamma_pdf(
            distances,
            concentration=gamma_concentration,
            rate=gamma_distance,
            scaling=None,
        )

        return sigmoid_distance_function + gamma_distance_function

    def inverse_sigmoid_signal_distance_function(
        self,
        distances,
        layer,
        weights_shape=None,
        name="signal_distance_",
        mode="independent_effect",
        average_distance_prior=None,
    ) -> torch.Tensor:
        if weights_shape is None:
            weights_shape = [self.n_signals]
        if average_distance_prior is None:
            average_distance_prior = self.average_distance_prior
        # returns weights for [n_signals, n_distance_bins]
        if distances.dim() == 1:
            distances = distances.unsqueeze(-2)
        elif distances.dim() == 2:
            distances = distances.unsqueeze(-3)
        return self.inverse_sigmoid_distance_function(
            distances,
            layer,
            weights_shape=weights_shape,
            name=name,
            mode=mode,
            average_distance_prior=average_distance_prior,
        )

    def forward(self, *args, **kwargs):
        return getattr(self, self.mode)(
            *args,
            **kwargs,
        )

    def signal_receptor_tf_effect(
        self,
        bound_receptor_abundance_src: torch.Tensor,
        use_cell_abundance_model: bool = False,
    ):
        layer = 0
        # optionally apply dropout ==========
        if self.dropout_rate > 0:
            if getattr(self.weights, f"{self.name}_layer_{layer}_dropout", None) is None:
                deep_setattr(
                    self.weights,
                    f"{self.name}_layer_{layer}_dropout",
                    nn.Dropout(p=self.dropout_rate),
                )
            dropout = deep_getattr(self.weights, f"{self.name}_layer_{layer}_dropout")
            bound_receptor_abundance_src = dropout(bound_receptor_abundance_src)

        # 3. Computing effects a_{r,s,h} of ligand-receptor complexes x_{c,r,s} on active TF concentration. How? ==========
        # Basal TF active concentration ==========
        name = "basal_TF_weights"
        basal_tf_weights = self.get_tf_effect(
            x=bound_receptor_abundance_src,
            name=name,
            layer=layer,
            weights_shape=[self.n_tfs] if self.n_out == 1 else [self.n_tfs, self.n_out],
            remove_diagonal=False,
        )
        # Signal-receptor complex effect on TF active concentration ==========
        tf_sig_rec_effect_hsr = self.get_signal_receptor_tf_effect(
            x=bound_receptor_abundance_src,
            layer=layer,
            weights_shape=[self.n_tfs, len(self.signal_receptor_mask_scipy.data)]
            if self.n_out == 1
            else [self.n_tfs, len(self.signal_receptor_mask_scipy.data), self.n_out],
            name="signal_receptor_tf_effect",
        )

        # print("tf_sig_rec_effect_hsr mean", tf_sig_rec_effect_hsr.mean())
        # print("tf_sig_rec_effect_hsr min", tf_sig_rec_effect_hsr.min())
        # print("tf_sig_rec_effect_hsr max", tf_sig_rec_effect_hsr.max())
        # print("bound_receptor_abundance_crs mean", bound_receptor_abundance_crs.mean())
        # print("bound_receptor_abundance_crs min", bound_receptor_abundance_crs.min())
        # print("bound_receptor_abundance_crs max", bound_receptor_abundance_crs.max())
        if self.n_out == 1:
            if use_cell_abundance_model:
                bound_rs = rearrange(
                    bound_receptor_abundance_src,
                    "r (c h) -> c h r",
                    h=self.n_tfs,
                )
                effect = torch.einsum("chr,hr->ch", bound_rs, tf_sig_rec_effect_hsr)
            else:
                effect = torch.einsum(
                    "cr,hr->ch",
                    bound_receptor_abundance_src.coalesce().values(),
                    tf_sig_rec_effect_hsr,
                )
            effect_on_tf_abundance = (
                # independent term for basal TF active concentration
                basal_tf_weights.unsqueeze(-2)
                # Communication-dependent effect on TF active concentration
                + effect
            )
        else:
            if use_cell_abundance_model:
                bound_rs = rearrange(
                    bound_receptor_abundance_src,
                    "r (c h) -> c h r",
                    h=self.n_tfs,
                )
                effect = torch.einsum("chr,hrf->fch", bound_rs, tf_sig_rec_effect_hsr)
            else:
                effect = torch.einsum(
                    "cr,hrf->fch",
                    bound_receptor_abundance_src.coalesce().values(),
                    tf_sig_rec_effect_hsr,
                )
            effect_on_tf_abundance = (
                # independent term for basal TF active concentration
                basal_tf_weights.T.unsqueeze(-2)
                # Communication-dependent effect on TF active concentration
                + effect
            )

        if self.n_pathways > 1:
            effect_on_tf_abundance = effect_on_tf_abundance + self.signal_receptor_pathway_tf_effect(
                bound_receptor_abundance_src=bound_receptor_abundance_src,
                use_cell_abundance_model=use_cell_abundance_model,
            )

        # print("self.output_transform", self.output_transform)
        # print("effect_on_tf_abundance mean", effect_on_tf_abundance.mean())
        # print("effect_on_tf_abundance min", effect_on_tf_abundance.min())
        # print("effect_on_tf_abundance max", effect_on_tf_abundance.max())
        if use_cell_abundance_model:
            effect_on_tf_abundance = effect_on_tf_abundance / torch.tensor(100.0, device=effect_on_tf_abundance.device)
        if self.output_transform == "softplus":
            # apply softplus to ensure positive values
            effect_on_tf_abundance = nn.functional.softplus(
                effect_on_tf_abundance / torch.tensor(1.0, device=effect_on_tf_abundance.device)
            ) / torch.tensor(0.7, device=effect_on_tf_abundance.device)
        elif self.output_transform == "proportion1":
            # The proportion of TF in the active form
            # TODO #  how do basal_tf_weights behave? average 2 before transform and the other effect is normalised to be small?
            effect_on_tf_abundance = torch.sigmoid(
                effect_on_tf_abundance / torch.tensor(1.0, device=effect_on_tf_abundance.device)
                + torch.tensor(1.0, device=effect_on_tf_abundance.device)
            )
        elif self.output_transform == "proportion0":
            # The proportion of TF in the active form
            # TODO #  how do basal_tf_weights behave? average 2 before transform and the other effect is normalised to be small?
            effect_on_tf_abundance = torch.sigmoid(
                effect_on_tf_abundance / torch.tensor(1.0, device=effect_on_tf_abundance.device)
                - torch.tensor(1.0, device=effect_on_tf_abundance.device)
            )
        elif self.output_transform == "activity":
            # NOTE that this does not use TF abundance info (similarly to independent TF activity term)
            # This represents effect direction but not magnitude
            effect_on_tf_abundance = torch.sigmoid(
                effect_on_tf_abundance / torch.tensor(1.0, device=effect_on_tf_abundance.device)
            ) * torch.tensor(2.0, device=effect_on_tf_abundance.device) - torch.tensor(
                1.0, device=effect_on_tf_abundance.device
            )

        # print(f"cell_comm_effect {self.name} mean ", tf_abundance.mean())
        # print(f"cell_comm_effect {self.name} min", tf_abundance.min())
        # print(f"cell_comm_effect {self.name} max", tf_abundance.max())

        return effect_on_tf_abundance

    def signal_receptor_pathway_tf_effect(
        self,
        bound_receptor_abundance_src: torch.Tensor,
        use_cell_abundance_model: bool = False,
    ):
        layer = 0

        # Basal pathway activity ==========
        name = "basal_pathway_weights"
        basal_pathway_weights = self.get_tf_effect(
            x=bound_receptor_abundance_src,
            name=name,
            layer=layer,
            weights_shape=[self.n_pathways] if self.n_out == 1 else [self.n_pathways, self.n_out],
            remove_diagonal=False,
        )
        # Signal-receptor complex effect on pathway activity ==========
        pathway_sig_rec_effect_hsr = self.get_signal_receptor_tf_effect(
            x=bound_receptor_abundance_src,
            layer=layer,
            weights_shape=[self.n_pathways, len(self.signal_receptor_mask_scipy.data)]
            if self.n_out == 1
            else [
                self.n_pathways,
                len(self.signal_receptor_mask_scipy.data),
                self.n_out,
            ],
            name="signal_receptor_pathway_effect",
        )
        # normalised by sqrt of the number of predictors
        pathway_sig_rec_effect_hsr = pathway_sig_rec_effect_hsr / torch.sqrt(
            torch.tensor(
                float(len(self.signal_receptor_mask_scipy.data)),
                device=bound_receptor_abundance_src.device,
            )
        )

        # print("pathway_sig_rec_effect_hsr mean", pathway_sig_rec_effect_hsr.mean())
        # print("pathway_sig_rec_effect_hsr min", pathway_sig_rec_effect_hsr.min())
        # print("pathway_sig_rec_effect_hsr max", pathway_sig_rec_effect_hsr.max())
        # print("bound_receptor_abundance_crs mean", bound_receptor_abundance_crs.mean())
        # print("bound_receptor_abundance_crs min", bound_receptor_abundance_crs.min())
        # print("bound_receptor_abundance_crs max", bound_receptor_abundance_crs.max())
        if self.n_out == 1:
            effect_on_pathway_activity = (
                # independent term for basal pathway activity
                basal_pathway_weights.unsqueeze(-2)
                # Communication-dependent effect on pathway activity
                + torch.einsum(
                    "rc,hr->ch",
                    bound_receptor_abundance_src,
                    pathway_sig_rec_effect_hsr,
                )
            )
        else:
            effect_on_pathway_activity = (
                # independent term for basal pathway activity
                basal_pathway_weights.T.unsqueeze(-2)
                # Communication-dependent effect on pathway activity
                + torch.einsum(
                    "rc,hrf->fch",
                    bound_receptor_abundance_src,
                    pathway_sig_rec_effect_hsr,
                )
            )
        # print("self.output_transform", self.output_transform)
        # print("effect_on_tf_abundance mean", effect_on_tf_abundance.mean())
        # print("effect_on_tf_abundance min", effect_on_tf_abundance.min())
        # print("effect_on_tf_abundance max", effect_on_tf_abundance.max())

        # apply softplus to ensure positive values
        effect_on_pathway_activity = nn.functional.softplus(
            effect_on_pathway_activity / torch.tensor(1.0, device=effect_on_pathway_activity.device)
        ) / torch.tensor(0.7, device=effect_on_pathway_activity.device)

        # compute pathway effects on TFs
        name = "pathway_tf_weights"
        pathway_tf_weights = self.get_tf_effect(
            x=bound_receptor_abundance_src,
            name=name,
            layer=layer,
            weights_shape=[self.n_tfs, self.n_pathways]
            if self.n_out == 1
            else [self.n_tfs, self.n_pathways * self.n_out],
            remove_diagonal=False,
            non_negative=self.use_non_negative_weights,
        )
        if use_cell_abundance_model:
            effect_on_pathway_activity = rearrange(
                effect_on_pathway_activity,
                "(c h) p -> c h p",
                h=self.n_tfs,
            )
        if self.n_out == 1:
            if use_cell_abundance_model:
                effect_on_tf_activity = torch.einsum("chp,hp->ch", effect_on_pathway_activity, pathway_tf_weights)
            else:
                effect_on_tf_activity = torch.einsum("cp,hp->ch", effect_on_pathway_activity, pathway_tf_weights)
        else:
            pathway_tf_weights = rearrange(pathway_tf_weights, "h (p f) -> h p f", p=self.n_pathways, f=self.n_out)
            if use_cell_abundance_model:
                effect_on_tf_activity = torch.einsum("chp,hpf->fch", effect_on_pathway_activity, pathway_tf_weights)
            else:
                effect_on_tf_activity = torch.einsum("cp,hpf->fch", effect_on_pathway_activity, pathway_tf_weights)
        # including pathway interactions
        if self.use_pathway_interaction_effect:
            name = "pathway_interaction_tf_weights"
            pathway_tf_weights = self.get_tf_effect(
                x=bound_receptor_abundance_src,
                name=name,
                layer=layer,
                weights_shape=[self.n_tfs, self.n_pathways * self.n_pathways]
                if self.n_out == 1
                else [self.n_tfs, self.n_pathways * self.n_pathways * self.n_out],
                remove_diagonal=False,
                non_negative=self.use_non_negative_weights,
            )
            if self.n_out == 1:
                pathway_tf_weights = rearrange(
                    pathway_tf_weights,
                    "h (o p) -> h o p",
                    p=self.n_pathways,
                    o=self.n_pathways,
                )
                if use_cell_abundance_model:
                    effect_on_tf_activity = effect_on_tf_activity + torch.einsum(
                        "chp,hop,cho->ch",
                        effect_on_pathway_activity,
                        pathway_tf_weights,
                        effect_on_pathway_activity,
                    )
                else:
                    effect_on_tf_activity = effect_on_tf_activity + torch.einsum(
                        "cp,hop,co->ch",
                        effect_on_pathway_activity,
                        pathway_tf_weights,
                        effect_on_pathway_activity,
                    )
            else:
                pathway_tf_weights = rearrange(
                    pathway_tf_weights,
                    "h (o p f) -> h o p f",
                    p=self.n_pathways,
                    o=self.n_pathways,
                    f=self.n_out,
                )
                if use_cell_abundance_model:
                    effect_on_tf_activity = torch.einsum(
                        "chp,hopf,cho->fch",
                        effect_on_pathway_activity,
                        pathway_tf_weights,
                        effect_on_pathway_activity,
                    )
                else:
                    effect_on_tf_activity = torch.einsum(
                        "cp,hopf,co->fch",
                        effect_on_pathway_activity,
                        pathway_tf_weights,
                        effect_on_pathway_activity,
                    )

        # print(f"effect_on_tf_activity pathway {self.name} mean ", effect_on_tf_activity.mean())
        # print(f"effect_on_tf_activity pathway {self.name} min", effect_on_tf_activity.min())
        # print(f"effect_on_tf_activity pathway {self.name} max", effect_on_tf_activity.max())

        return effect_on_tf_activity

    def signal_receptor_occupancy(
        self,
        signal_abundance: torch.Tensor,
        receptor_abundance: torch.Tensor,
        distances: torch.Tensor = None,
        skip_distance_effect: bool = False,
    ):
        layer = 0
        # optionally apply dropout ==========
        if self.dropout_rate > 0:
            if getattr(self.weights, f"{self.name}_layer_{layer}_dropout", None) is None:
                deep_setattr(
                    self.weights,
                    f"{self.name}_layer_{layer}_dropout",
                    nn.Dropout(p=self.dropout_rate),
                )
            dropout = deep_getattr(self.weights, f"{self.name}_layer_{layer}_dropout")
            signal_abundance = dropout(signal_abundance)
            receptor_abundance = dropout(receptor_abundance)

        # 1. Signal RNA -> signal protein conversion using distance function ============
        # a_{s, b} = f(signal_protein_features, distance_between_bins)
        # w_{c,s} = sum_b w_{c,s,b} * a_{s, b}
        if not skip_distance_effect:
            signal_distance_effect_sb = self.inverse_sigmoid_signal_distance_function(
                distances,
                layer=layer,
            )
            signal_abundance = torch.einsum("csb,sb->cs", signal_abundance, signal_distance_effect_sb)

        # 2. Computing bound receptor concentrations using learnable a_{r,s} affinity ============
        # a_{r,s} = f(receptor_features, signal_features)
        sig_rec_affinity_rs = self.get_signal_receptor_effect(
            x=signal_abundance,
            layer=layer,
            weights_shape=[len(self.signal_receptor_mask_scipy.data)],
        )

        # x_{c,r,s} = w_{c,s} * a_{r,s}
        row2signal = one_hot(
            torch.tensor(self.signal_receptor_mask_scipy.col, device=signal_abundance.device).long().unsqueeze(-1),
            self.signal_receptor_mask_scipy.shape[1],
        )
        affinity_to_receptor_src = signal_abundance.T[
            self.signal_receptor_mask_scipy.row, :
        ] * sig_rec_affinity_rs.unsqueeze(-1)
        affinity_to_receptor_rc_sum = torch.mm(row2signal.T, affinity_to_receptor_src)
        # affinity_to_receptor_crs = torch.einsum(
        #    "cs,rs->crs", signal_abundance, sig_rec_affinity_rs
        # )
        # Compute bound receptor abundance
        # bound = total * proportion_of_signal_with_affinity
        # TODO - which unbound term to use?
        # x_{c,r,s} = w_{c,r} * (x_{c,r,s} / (sum_s x_{c,r,s} + unbound_r))
        # x_{c,r,s} = w_{c,r} * (x_{c,r,s} / (sum_s x_{c,r,s} + unbound_r * w_{c,r}))
        unbound_r = self.get_signal_distance_effect(
            x=signal_abundance,
            layer=layer,
            name="unbound_r",
            weights_shape=[self.n_receptors],
        )
        unbound_r = nn.functional.softplus(
            unbound_r / torch.tensor(5.0, device=unbound_r.device) - torch.tensor(2.0, device=unbound_r.device)
        )
        if not self.use_unbound_concentration:
            proportion_of_signal_with_affinity_src = affinity_to_receptor_src / (
                affinity_to_receptor_rc_sum[self.signal_receptor_mask_scipy.col, :]
                + unbound_r[self.signal_receptor_mask_scipy.col].unsqueeze(-1)
            )
        else:
            proportion_of_signal_with_affinity_src = affinity_to_receptor_src / (
                affinity_to_receptor_rc_sum[self.signal_receptor_mask_scipy.col, :]
                + unbound_r[self.signal_receptor_mask_scipy.col].unsqueeze(-1)
                * receptor_abundance.T[self.signal_receptor_mask_scipy.col, :]
            )
        bound_receptor_abundance_src = (
            proportion_of_signal_with_affinity_src * receptor_abundance.T[self.signal_receptor_mask_scipy.col, :]
        )
        # bound_receptor_abundance_crs = torch.einsum(
        #    "cr,crs->crs", receptor_abundance, proportion_of_signal_with_affinity_src
        # )
        # optionally apply dropout
        # if self.dropout_rate > 0:
        #     bound_receptor_abundance_crs = dropout(bound_receptor_abundance_crs)

        return bound_receptor_abundance_src

    def signal_receptor_occupancy_spatial(
        self,
        signal_abundance: torch.Tensor,
        receptor_abundance: torch.Tensor,
        distances: torch.Tensor = None,
        obs_plate=None,
    ):
        n_locations = signal_abundance.shape[-2]
        n_signals = signal_abundance.shape[-1]
        n_receptors = receptor_abundance.shape[-1]
        n_cell_types = receptor_abundance.shape[-2]

        if distances.is_sparse:
            # with obs_plate as ind:
            #    pass
            # indices0 = distances.coalesce().indices()[0, :]
            indices1 = distances.coalesce().indices()[1, :]
            distances_ = distances.coalesce().values()
            # indices = torch.logical_or(torch.isin(indices0, ind), torch.isin(indices1, ind))
            # indices0 = indices0[indices]
            # use 1d tensor with propper indices mapping here to make sure that the indices are correct
            # indices1 = indices1[indices]
            # distances_ = distances_[indices]

            # 1. Signal RNA -> signal protein conversion using distance function ============
            signal_distance_effect_ss_b = self.inverse_sigmoid_signal_distance_function(
                distances_,
                layer="0",
                name="signal_distance_spatial_",
            ).T
            target2row = one_hot(
                torch.as_tensor(indices1, device=signal_abundance.device).long().unsqueeze(-1),
                distances.shape[1],
            ).T
            signal_abundance = torch.mm(
                target2row,
                signal_distance_effect_ss_b * signal_abundance[indices1, :],  # target s to row  # row to signal
            )
        else:
            # 1. Signal RNA -> signal protein conversion using distance function ============
            signal_distance_effect_ss_b = self.inverse_sigmoid_signal_distance_function(
                distances,
                layer="0",
                name="signal_distance_spatial_",
            )
            signal_abundance = torch.einsum("ps,sop->os", signal_abundance, signal_distance_effect_ss_b)

        # 2. Computing bound receptor concentrations using learnable a_{r,s} affinity ============
        # first reshape inputs to be locations * cell type specific
        # d_{c,s} -> d_{c,f,s}
        signal_abundance = signal_abundance.unsqueeze(-2).expand([n_locations, n_cell_types, n_signals])
        signal_abundance = rearrange(signal_abundance, "c f s -> (c f) s", f=n_cell_types)
        # g_{f,r} -> g_{c,f,r}
        receptor_abundance = receptor_abundance.unsqueeze(-3).expand([n_locations, n_cell_types, n_receptors])
        receptor_abundance = rearrange(receptor_abundance, "c f r -> (c f) r", f=n_cell_types)

        bound_receptor_abundance_src = self.signal_receptor_occupancy(
            signal_abundance=signal_abundance,
            receptor_abundance=receptor_abundance,
            distances=distances,
            skip_distance_effect=True,
        )
        return bound_receptor_abundance_src

    def signal_receptor_tf_effect_spatial(
        self,
        bound_receptor_abundance_src: torch.Tensor,
    ):
        return self.signal_receptor_tf_effect(
            bound_receptor_abundance_src=bound_receptor_abundance_src,
            use_cell_abundance_model=True,
        )
