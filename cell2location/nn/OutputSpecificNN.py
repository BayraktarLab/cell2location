from typing import Iterable

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr
from pyro.nn import PyroModule, PyroParam
from torch import nn as nn
from torch.distributions import constraints


class OutputSpecificNN(PyroModule):
    """
    Model which defines small, output dimension-specific NNs. Inspired by DCDI and scvi-tools FCLayers.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    use_activation
        Whether to have layer activation at last layer or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        name: str = "",
        n_out_extra: int = 1,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 16,
        dropout_rate: float = 0.1,
        bayesian: bool = False,
        use_non_negative_weights: bool = False,
        use_layer_norm: bool = True,
        use_last_layer_norm: bool = False,
        use_activation: bool = True,
        use_last_activation: bool = False,
        use_global_weights: bool = False,
        bias: bool = True,
        last_bias: bool = False,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ELU,
        weights_prior={"shape": 0.1, "scale": 1.0},
        bias_prior={"mean": -10.0, "sigma": 3.0},
    ):
        super().__init__()

        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        self.n_out_extra = n_out_extra
        self.n_cat_list = n_cat_list
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate
        self.bayesian = bayesian
        self.use_non_negative_weights = use_non_negative_weights
        self.use_layer_norm = use_layer_norm
        self.use_last_layer_norm = use_last_layer_norm
        self.use_activation = use_activation
        self.use_last_activation = use_last_activation
        self.use_global_weights = use_global_weights
        self.bias = bias
        self.last_bias = last_bias
        self.inject_covariates = inject_covariates
        self.activation_fn = activation_fn
        self.weights_prior = weights_prior
        self.bias_prior = bias_prior

        self.weights = PyroModule()

        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("zeros", torch.zeros((1, 1)))
        self.register_buffer("weights_prior_shape", torch.tensor(self.weights_prior["shape"]))
        self.register_buffer("weights_prior_scale", torch.tensor(self.weights_prior["scale"]))
        self.register_buffer("bias_mean_prior", torch.tensor(self.bias_prior["mean"]))
        self.register_buffer("bias_sigma_prior", torch.tensor(self.bias_prior["sigma"]))

    def forward(
        self,
        x: torch.Tensor,
        *in_out_effect,
    ):

        for layer in range(self.n_layers + 1):
            if self.use_global_weights and (layer < self.n_layers):
                n_out = 1
            else:
                n_out = self.n_out
            if layer == 0:
                n_in = self.n_in
            else:
                n_in = self.n_hidden
            if layer == self.n_layers:
                n_hidden = self.n_out_extra
            else:
                n_hidden = self.n_hidden
            # optionally apply dropout ==========
            if self.dropout_rate > 0:
                if getattr(self.weights, f"{self.name}_layer_{layer}_dropout", None) is None:
                    deep_setattr(
                        self.weights,
                        f"{self.name}_layer_{layer}_dropout",
                        nn.Dropout(p=self.dropout_rate),
                    )
                dropout = deep_getattr(self.weights, f"{self.name}_layer_{layer}_dropout")
                x = dropout(x)

            # generate parameters ==========
            if self.bayesian:
                # generate bayesian variables
                if self.use_non_negative_weights:
                    # define Gamma distributed weights and Normal bias
                    # positive effect of input on output
                    weights = pyro.sample(
                        f"{self.name}_weights_layer_{layer}",
                        # for every TF increase alpha
                        # (more TFs have less sparse distribution)
                        dist.Gamma(
                            self.weights_prior_shape.to(x.device),
                            self.ones.to(x.device),
                        )
                        .expand([n_hidden, n_out, n_in])
                        .to_event(3),
                    )
                else:
                    # define laplace prior or horseshoe prior TODO horseshoe
                    weights = pyro.sample(
                        f"{self.name}_weights_layer_{layer}",
                        # for every TF increase alpha
                        # (more TFs have less sparse distribution)
                        dist.Laplace(
                            self.zeros.to(x.device),
                            self.weights_prior_scale.to(x.device),
                        )
                        .expand([n_hidden, n_out, n_in])
                        .to_event(3),
                    )

                # bias allows requiring signal from more than one input
                bias = pyro.sample(
                    f"{self.name}_bias_layer_{layer}",
                    dist.Normal(
                        self.bias_mean_prior.to(x.device),
                        self.ones.to(x.device) * self.bias_sigma_prior.to(x.device),
                    )
                    .expand([1, n_out, n_hidden])
                    .to_event(3),
                )
            else:
                if getattr(self.weights, f"{self.name}_layer_{layer}_weights", None) is None:
                    if self.use_non_negative_weights:
                        # initialise weights
                        init_param = torch.normal(
                            torch.full(
                                size=(n_hidden, n_out, n_in),
                                fill_value=0.0,
                                device=x.device,
                            ),
                            torch.full(
                                size=(n_hidden, n_out, n_in),
                                fill_value=1 / np.sqrt(n_hidden + n_out),
                                device=x.device,
                            ),
                        ).abs()
                        deep_setattr(
                            self.weights,
                            f"{self.name}_layer_{layer}_weights",
                            PyroParam(
                                init_param.clone().detach().requires_grad_(True),
                                constraint=constraints.positive,
                            ),
                        )
                        init_param = torch.normal(
                            torch.full(
                                size=(1, n_out, n_hidden),
                                fill_value=0.0,
                                device=x.device,
                            ),
                            torch.full(
                                size=(1, n_out, n_hidden),
                                fill_value=1 / np.sqrt(n_hidden + n_out),
                                device=x.device,
                            ),
                        )
                        deep_setattr(
                            self.weights,
                            f"{self.name}_layer_{layer}_bias",
                            PyroParam(
                                init_param.clone().detach().requires_grad_(True),
                            ),
                        )
                    else:
                        # initialise weights
                        init_param = torch.normal(
                            torch.full(
                                size=(n_hidden, n_out, n_in),
                                fill_value=0.0,
                                device=x.device,
                            ),
                            torch.full(
                                size=(n_hidden, n_out, n_in),
                                fill_value=1 / np.sqrt(n_hidden + n_out),
                                device=x.device,
                            ),
                        )
                        deep_setattr(
                            self.weights,
                            f"{self.name}_layer_{layer}_weights",
                            PyroParam(init_param.clone().detach().requires_grad_(True)),
                        )

                        init_param = torch.normal(
                            torch.full(
                                size=(1, n_out, n_hidden),
                                fill_value=0.0,
                                device=x.device,
                            ),
                            torch.full(
                                size=(1, n_out, n_hidden),
                                fill_value=1 / np.sqrt(n_hidden + n_out),
                                device=x.device,
                            ),
                        )
                        deep_setattr(
                            self.weights,
                            f"{self.name}_layer_{layer}_bias",
                            PyroParam(init_param.clone().detach().requires_grad_(True)),
                        )

                # extract weights
                weights = deep_getattr(self.weights, f"{self.name}_layer_{layer}_weights")
                bias = deep_getattr(self.weights, f"{self.name}_layer_{layer}_bias")
                if self.use_global_weights:
                    weights = weights.expand([n_hidden, self.n_out, n_in])
                    bias = bias.expand([1, self.n_out, n_hidden])

            # compute layer weighted sum using einsum ==========
            if (len(in_out_effect) == 1) and (layer == 0):
                # first layer, apply in_out_effect (fg)
                if len(in_out_effect[0].shape) == 2:
                    in_out_effect = in_out_effect[0].unsqueeze(0)
                    x = torch.einsum("tfg,lfg,cg->cft", weights, in_out_effect, x)
                elif len(in_out_effect[0].shape) == 3:
                    x = torch.einsum("tfg,cfg,cg->cft", weights, in_out_effect[0], x)
            elif (len(in_out_effect) == 0) and (layer == 0):
                # first layer, without in_out_effect (fg)
                x = torch.einsum("tfg,cg->cft", weights, x)
            else:
                # second layer or more
                x = torch.einsum("qft,cft->cfq", weights, x)
            if ((layer < self.n_layers) and self.bias) or ((layer == self.n_layers) and self.last_bias and self.bias):
                x = x + bias

            # optionally apply layernorm ==========
            if ((layer < self.n_layers) and self.use_layer_norm) or (
                (layer == self.n_layers) and self.use_last_layer_norm and self.use_layer_norm
            ):
                if getattr(self.weights, f"{self.name}_layer_{layer}_layer_norm", None) is None:
                    deep_setattr(
                        self.weights,
                        f"{self.name}_layer_{layer}_layer_norm",
                        nn.LayerNorm((self.n_out, n_hidden), elementwise_affine=False),
                    )
                layer_norm = deep_getattr(self.weights, f"{self.name}_layer_{layer}_layer_norm")
                x = layer_norm(x)

            # optionally apply activation ==========
            if ((layer < self.n_layers) and self.use_activation) or (
                (layer == self.n_layers) and self.use_last_activation and self.use_activation
            ):
                if getattr(self.weights, f"{self.name}_layer_{layer}_activation_fn", None) is None:
                    deep_setattr(
                        self.weights,
                        f"{self.name}_layer_{layer}_activation_fn",
                        self.activation_fn(),
                    )
                activation_fn = deep_getattr(self.weights, f"{self.name}_layer_{layer}_activation_fn")
                x = activation_fn(x)
        if self.n_out_extra == 1:
            x = x.squeeze(-1)
        return x
