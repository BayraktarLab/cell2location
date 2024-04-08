import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr
from pyro.nn import PyroParam, PyroSample
from torch import nn as nn
from torch.distributions import constraints


class CreateParameterMixin:
    def create_horseshoe_prior(
        self,
        name,
        weights_shape,
        weights_prior_scale=None,
        weights_prior_tau=None,
        scale_distribution=dist.HalfNormal,  # TODO figure out which distribution to use HalfCauchy has mean=Inf so can't use it
    ):
        # Create scalar tau (like sd for horseshoe prior) =====================
        tau_name = f"{name}tau"
        if getattr(self.weights, tau_name, None) is None:
            if weights_prior_tau is None:
                weights_prior_tau = self.weights_prior_tau
            if getattr(self, f"{tau_name}_scale", None) is None:
                self.register_buffer(f"{tau_name}_scale", weights_prior_tau)
            deep_setattr(
                self.weights,
                tau_name,
                PyroSample(
                    lambda prior: scale_distribution(
                        getattr(self, f"{tau_name}_scale"),
                    )
                    .expand([1])
                    .to_event(1),
                ),
            )
        tau = deep_getattr(self.weights, tau_name)

        # Create weights (like mean for horseshoe prior) =====================
        weights_name = f"{name}weights"
        if getattr(self.weights, weights_name, None) is None:
            deep_setattr(
                self.weights,
                weights_name,
                PyroSample(
                    lambda prior: dist.Normal(
                        self.zeros,
                        self.ones,
                    )
                    .expand(weights_shape)
                    .to_event(len(weights_shape)),
                ),
            )
        unscaled_weights = deep_getattr(self.weights, weights_name)

        if getattr(self, "use_gamma_horseshoe_prior", False):
            # Create elementwise lambdas using Gamma distribution (like sd for horseshoe prior) =====================
            lambdas_name = f"{name}lambdas"
            if getattr(self.weights, lambdas_name, None) is None:
                if weights_prior_scale is None:
                    weights_prior_scale = self.weights_prior_scale
                if getattr(self, f"{lambdas_name}_scale", None) is None:
                    self.register_buffer(f"{lambdas_name}_scale", weights_prior_scale)
                deep_setattr(
                    self.weights,
                    lambdas_name,
                    PyroSample(
                        lambda prior: dist.Gamma(
                            tau,
                            getattr(self, f"{lambdas_name}_scale"),
                        )
                        .expand(weights_shape)
                        .to_event(len(weights_shape)),
                    ),
                )
            lambdas = deep_getattr(self.weights, lambdas_name)
        else:
            # Create elementwise lambdas (like sd for horseshoe prior) =====================
            lambdas_name = f"{name}lambdas"
            if getattr(self.weights, lambdas_name, None) is None:
                if weights_prior_scale is None:
                    weights_prior_scale = self.weights_prior_scale
                if getattr(self, f"{lambdas_name}_scale", None) is None:
                    self.register_buffer(f"{lambdas_name}_scale", weights_prior_scale)
                deep_setattr(
                    self.weights,
                    lambdas_name,
                    PyroSample(
                        lambda prior: scale_distribution(
                            getattr(self, f"{lambdas_name}_scale"),
                        )
                        .expand(weights_shape)
                        .to_event(len(weights_shape)),
                    ),
                )
            lambdas = deep_getattr(self.weights, lambdas_name)
            lambdas = tau * lambdas

        weights = lambdas * unscaled_weights
        if not self.training:
            pyro.deterministic(f"{self.name_prefix}{name}", weights)
        return weights

    def get_param(
        self,
        x: torch.Tensor,
        name: str,
        layer,
        weights_shape: list,
        random_init_scale: float = 1.0,
        bayesian: bool = True,
        use_non_negative_weights: bool = False,
        bias_shape: list = [1],
        skip_name: bool = False,
        weights_prior_mean: torch.Tensor = None,
        weights_prior_scale: torch.Tensor = None,
        weights_prior_shape: torch.Tensor = None,
        weights_prior_rate: torch.Tensor = None,
        weights_prior_tau: torch.Tensor = None,
        return_bias: bool = False,
        use_horseshoe_prior: bool = False,
    ):
        # generate parameter names ==========
        if skip_name:
            weights_name = f"{name}_layer_{layer}_weights"
            bias_name = f"{name}_layer_{layer}_bias"
        else:
            weights_name = f"{self.name}_{name}_layer_{layer}_weights"
            bias_name = f"{self.name}_{name}_layer_{layer}_bias"

        # create parameters ==========
        if not use_horseshoe_prior:
            # register priors ==========
            if weights_prior_mean is None:
                weights_prior_mean = self.zeros
            if getattr(self, f"{weights_name}_mean", None) is None:
                self.register_buffer(f"{weights_name}_mean", weights_prior_mean)
            if weights_prior_scale is None:
                weights_prior_scale = self.weights_prior_scale
            if getattr(self, f"{weights_name}_scale", None) is None:
                self.register_buffer(f"{weights_name}_scale", weights_prior_scale)
            if weights_prior_shape is None:
                weights_prior_shape = self.weights_prior_shape
            if getattr(self, f"{weights_name}_shape", None) is None:
                self.register_buffer(f"{weights_name}_shape", weights_prior_shape)
            if weights_prior_rate is None:
                weights_prior_rate = self.ones
            if getattr(self, f"{weights_name}_rate", None) is None:
                self.register_buffer(f"{weights_name}_rate", weights_prior_rate)
            # create parameters ==========
            if getattr(self.weights, weights_name, None) is None:
                if bayesian:
                    # generate bayesian variables
                    if use_non_negative_weights:
                        # define Gamma distributed weights and Normal bias
                        # positive effect of input on output
                        deep_setattr(
                            self.weights,
                            weights_name,
                            PyroSample(
                                lambda prior: dist.Gamma(
                                    getattr(self, f"{weights_name}_shape"),
                                    getattr(self, f"{weights_name}_rate"),
                                )
                                .expand(weights_shape)
                                .to_event(len(weights_shape))
                            ),
                        )
                    else:
                        deep_setattr(
                            self.weights,
                            weights_name,
                            PyroSample(
                                lambda prior: dist.SoftLaplace(
                                    getattr(self, f"{weights_name}_mean"),
                                    getattr(self, f"{weights_name}_scale"),
                                )
                                .expand(weights_shape)
                                .to_event(len(weights_shape)),
                            ),
                        )
                    if return_bias:
                        # bias allows requiring signal from more than one input
                        deep_setattr(
                            self.weights,
                            bias_name,
                            PyroSample(
                                lambda prior: dist.Normal(
                                    self.bias_mean_prior,
                                    self.ones * self.bias_sigma_prior,
                                )
                                .expand(bias_shape)
                                .to_event(len(bias_shape)),
                            ),
                        )
                else:
                    if use_non_negative_weights:
                        # initialise weights
                        init_param = torch.normal(
                            torch.full(
                                size=weights_shape,
                                fill_value=0.0,
                                device=x.device,
                            ),
                            torch.full(
                                size=weights_shape,
                                fill_value=random_init_scale,
                                device=x.device,
                            ),
                        ).abs()
                        deep_setattr(
                            self.weights,
                            weights_name,
                            PyroParam(
                                init_param.clone().detach().requires_grad_(True),
                                constraint=constraints.positive,
                            ),
                        )
                    else:
                        # initialise weights
                        init_param = torch.normal(
                            torch.full(
                                size=weights_shape,
                                fill_value=0.0,
                                device=x.device,
                            ),
                            torch.full(
                                size=weights_shape,
                                fill_value=random_init_scale,
                                device=x.device,
                            ),
                        )
                        deep_setattr(
                            self.weights,
                            weights_name,
                            PyroParam(init_param.clone().detach().requires_grad_(True)),
                        )
                    if return_bias:
                        init_param = torch.normal(
                            torch.full(
                                size=bias_shape,
                                fill_value=0.0,
                                device=x.device,
                            ),
                            torch.full(
                                size=bias_shape,
                                fill_value=random_init_scale,
                                device=x.device,
                            ),
                        )
                        deep_setattr(
                            self.weights,
                            bias_name,
                            PyroParam(init_param.clone().detach().requires_grad_(True)),
                        )
            # extract parameters ==========
            weights = deep_getattr(self.weights, weights_name)
            if return_bias:
                bias = deep_getattr(self.weights, bias_name)
                return weights, bias
            return weights
        else:
            # create and extract parameters ==========
            return self.create_horseshoe_prior(
                name=weights_name,
                weights_shape=weights_shape,
                weights_prior_scale=weights_prior_scale,
                weights_prior_tau=weights_prior_tau,
            )

    def get_layernorm(self, name, layer, norm_shape):
        if getattr(self.weights, f"{self.name}_{name}_layer_{layer}_layer_norm", None) is None:
            deep_setattr(
                self.weights,
                f"{self.name}_{name}_layer_{layer}_layer_norm",
                nn.LayerNorm(norm_shape, elementwise_affine=False),
            )
        layer_norm = deep_getattr(self.weights, f"{self.name}_{name}_layer_{layer}_layer_norm")
        return layer_norm

    def get_activation(self, name, layer):
        if getattr(self.weights, f"{self.name}_{name}_layer_{layer}_activation_fn", None) is None:
            deep_setattr(
                self.weights,
                f"{self.name}_{name}_layer_{layer}_activation_fn",
                self.activation_fn(),
            )
        activation_fn = deep_getattr(self.weights, f"{self.name}_{name}_layer_{layer}_activation_fn")
        return activation_fn

    def get_pool(self, name, layer, kernel_size, pool_class=torch.nn.MaxPool2d):
        if getattr(self.weights, f"{self.name}_{name}_layer_{layer}_Pool", None) is None:
            deep_setattr(
                self.weights,
                f"{self.name}_{name}_layer_{layer}_Pool",
                pool_class(kernel_size),
            )
        max_pool = deep_getattr(self.weights, f"{self.name}_{name}_layer_{layer}_Pool")
        return max_pool

    def get_nn_weight(self, weights_name, weights_shape):
        if not hasattr(self.weights, weights_name):
            deep_setattr(
                self.weights,
                weights_name,
                pyro.nn.PyroSample(
                    lambda prior: dist.SoftLaplace(
                        self.zeros,
                        self.ones,
                    )
                    .expand(weights_shape)
                    .to_event(len(weights_shape)),
                ),
            )
        return deep_getattr(self.weights, weights_name)

    def get_nn_bias(self, bias_name, bias_shape):
        if not hasattr(self.weights, bias_name):
            deep_setattr(
                self.weights,
                bias_name,
                pyro.nn.PyroSample(
                    lambda prior: dist.SoftLaplace(
                        self.zeros,
                        self.ones,
                    )
                    .expand(bias_shape)
                    .to_event(len(bias_shape)),
                ),
            )
        return deep_getattr(self.weights, bias_name)

    def get_nn_layernorm(self, name, layer, norm_shape):
        if getattr(self.weights, f"{name}_layer_{layer}_layer_norm", None) is None:
            deep_setattr(
                self.weights,
                f"{name}_layer_{layer}_layer_norm",
                torch.nn.LayerNorm(norm_shape, elementwise_affine=False),
            )
        layer_norm = deep_getattr(self.weights, f"{name}_layer_{layer}_layer_norm")
        return layer_norm

    def get_nn_activation(self, name, layer):
        if getattr(self.weights, f"{name}_layer_{layer}_activation_fn", None) is None:
            deep_setattr(
                self.weights,
                f"{name}_layer_{layer}_activation_fn",
                torch.nn.Softplus(),
            )
        activation_fn = deep_getattr(self.weights, f"{name}_layer_{layer}_activation_fn")
        return activation_fn
