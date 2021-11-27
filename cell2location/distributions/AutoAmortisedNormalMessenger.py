from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
import pyro.distributions as dist
import torch
from pyro.distributions.distribution import Distribution
from pyro.distributions.transforms import SoftplusTransform
from pyro.infer.autoguide import AutoHierarchicalNormalMessenger
from pyro.infer.autoguide.utils import (
    deep_getattr,
    deep_setattr,
    helpful_support_errors,
)
from pyro.nn.module import PyroModule, PyroParam, to_pyro_module_
from scvi._compat import Literal
from scvi.nn import FCLayers
from torch.distributions import biject_to, constraints


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


class FCLayersPyro(FCLayers, PyroModule):
    pass


class AutoAmortisedHierarchicalNormalMessenger(AutoHierarchicalNormalMessenger):
    """
    EXPERIMENTAL Automatic :class:`~pyro.infer.effect_elbo.GuideMessenger` ,
    intended for use with :class:`~pyro.infer.effect_elbo.Effect_ELBO` or
    similar. Amortise specific sites

    The mean-field posterior at any site is a transformed normal distribution,
    the mean of which depends on the value of that site given its dependencies in the model:

        loc = loc + transform.inv(prior.mean) * weight

    Where the value of `prior.mean` is conditional on upstream sites in the model.
    This approach doesn't work for distributions that don't have the mean.

    loc, scales and element-specific weight are amortised for each site specified in `amortised_plate_sites`.

    Derived classes may override particular sites and use this simply as a
    default, see AutoNormalMessenger documentation for example.

    :param callable model: A Pyro model.
    :param dict amortised_plate_sites: Dictionary with amortised plate details:
        the name of observation/minibatch plate,
        indexes of model args to provide to encoder,
        variable names that belong to the observation plate
        and the number of dimensions in non-plate axis of each variable - such as:
        {
            "name": "obs_plate",
            "input": [0],  # expression data + (optional) batch index ([0, 2])
            "input_transform": [torch.log1p], # how to transform input data before passing to NN
            "sites": {
                "n_s": 1,
                "y_s": 1,
                "z_sr": R,
                "w_sf": F,
            }
        }
    :param int n_in: Number of input dimensions (for encoder_class).
    :param int n_hidden: Number of hidden nodes in each layer, one of 3 options:
        1. Integer denoting the number of hidden nodes
        2. Dictionary with {"single": 200, "multiple": 200} denoting the number of hidden nodes for each `encoder_mode` (See below)
        3. Allowing different number of hidden nodes for each model site. Dictionary with the number of hidden nodes for single encode mode and each model site:
        {
            "single": 200
            "n_s": 5,
            "y_s": 5,
            "z_sr": 128,
            "w_sf": 200,
        }
    :param float init_param_scale: How to scale/normalise initial values for weights converting hidden layers to loc and scales.
    :param float scales_offset: offset between the output of the NN and scales.
    :param Callable encoder_class: Class that defines encoder network.
    :param dict encoder_kwargs: Keyword arguments for encoder class.
    :param dict multi_encoder_kwargs: Optional separate keyword arguments for encoder_class,
        useful when encoder_mode == "single-multiple".
    :param Callable encoder_instance: Encoder network instance, overrides class input and the input instance is copied with deepcopy.
    :param str encoder_mode: Use single encoder for all variables ("single"), one encoder per variable ("multiple")
        or a single encoder in the first step and multiple encoders in the second step ("single-multiple").
    :param list hierarchical_sites: List of latent variables (model sites)
        that have hierarchical dependencies.
        If None, all sites are assumed to have hierarchical dependencies. If None, for the sites
        that don't have upstream sites, the guide is representing/learning deviation from the prior.
    """

    # 'element-wise' or 'scalar'
    weight_type = "element-wise"

    def __init__(
        self,
        model: Callable,
        *,
        amortised_plate_sites: dict,
        n_in: int,
        n_hidden: dict = None,
        init_param_scale: float = 1 / 50,
        init_scale: float = 0.1,
        init_weight: float = 1.0,
        encoder_class=FCLayersPyro,
        encoder_kwargs=None,
        multi_encoder_kwargs=None,
        encoder_instance: torch.nn.Module = None,
        encoder_mode: Literal["single", "multiple", "single-multiple"] = "single",
        hierarchical_sites: Optional[list] = None,
    ):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        super().__init__(model)
        self._init_scale = init_scale
        self._init_weight = init_weight
        self._hierarchical_sites = hierarchical_sites
        self.amortised_plate_sites = amortised_plate_sites
        self.encoder_mode = encoder_mode
        self._computing_median = False
        self._computing_mi = False

        self.softplus = SoftplusTransform()

        # default n_hidden values and checking input
        if n_hidden is None:
            n_hidden = {"single": 200, "multiple": 200}
        else:
            if isinstance(n_hidden, int):
                n_hidden = {"single": n_hidden, "multiple": n_hidden}
            elif not isinstance(n_hidden, dict):
                raise ValueError("n_hidden must be either int or dict")
        # process encoder kwargs, add n_hidden, create argument for multiple encoders
        encoder_kwargs = deepcopy(encoder_kwargs) if isinstance(encoder_kwargs, dict) else dict()
        encoder_kwargs["n_hidden"] = n_hidden["single"]
        if multi_encoder_kwargs is None:
            multi_encoder_kwargs = deepcopy(encoder_kwargs)

        # save encoder parameters
        self.encoder_kwargs = encoder_kwargs
        self.multi_encoder_kwargs = multi_encoder_kwargs
        self.single_n_in = n_in
        self.multiple_n_in = n_in
        self.n_hidden = n_hidden
        if ("single" in encoder_mode) and ("multiple" in encoder_mode):
            # if single network precedes multiple networks
            self.multiple_n_in = self.n_hidden["single"]
        self.encoder_class = encoder_class
        self.encoder_instance = encoder_instance
        self.init_param_scale = init_param_scale

    def get_posterior(
        self,
        name: str,
        prior: Distribution,
    ) -> Union[Distribution, torch.Tensor]:
        if self._computing_median:
            return self._get_posterior_median(name, prior)
        if self._computing_mi:
            return self._get_mutual_information(name, prior)

        with helpful_support_errors({"name": name, "fn": prior}):
            transform = biject_to(prior.support)
        # If hierarchical_sites not specified all sites are assumed to be hierarchical
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
            posterior = dist.TransformedDistribution(
                dist.Normal(loc, scale).to_event(transform.domain.event_dim),
                transform.with_cache(),
            )
            return posterior
        else:
            # Fall back to mean field when hierarchical_sites list is not empty and site not in the list.
            loc, scale = self._get_params(name, prior)
            posterior = dist.TransformedDistribution(
                dist.Normal(loc, scale).to_event(transform.domain.event_dim),
                transform.with_cache(),
            )
            return posterior

    def encode(self, name: str, prior: Distribution):
        """
        Apply encoder network to input data to obtain hidden layer encoding.
        Parameters
        ----------
        args
            Pyro model args
        kwargs
            Pyro model kwargs
        -------

        """
        try:
            args, kwargs = self.args_kwargs  # stored as a tuple of (tuple, dict)
            # get the data for NN from
            in_names = self.amortised_plate_sites["input"]
            x_in = [kwargs[i] if i in kwargs.keys() else args[i] for i in in_names]
            # apply data transform before passing to NN
            in_transforms = self.amortised_plate_sites["input_transform"]
            x_in = [in_transforms[i](x) for i, x in enumerate(x_in)]
            if "single" in self.encoder_mode:
                # encode with a single encoder
                res = deep_getattr(self, "one_encoder")(*x_in)
                if "multiple" in self.encoder_mode:
                    # when there is a second layer of multiple encoders fetch encoders and encode data
                    x_in[0] = res
                    res = deep_getattr(self.multiple_encoders, name)(*x_in)
            else:
                # when there are multiple encoders fetch encoders and encode data
                res = deep_getattr(self.multiple_encoders, name)(*x_in)
            return res
        except AttributeError:
            pass

        # Initialize.
        # create single encoder NN
        if "single" in self.encoder_mode:
            if self.encoder_instance is not None:
                # copy provided encoder instance
                one_encoder = deepcopy(self.encoder_instance).to(prior.mean.device)
                # convert to pyro module
                to_pyro_module_(one_encoder)
                deep_setattr(self, "one_encoder", one_encoder)
            else:
                # create encoder instance from encoder class
                deep_setattr(
                    self,
                    "one_encoder",
                    self.encoder_class(n_in=self.single_n_in, n_out=self.n_hidden["single"], **self.encoder_kwargs).to(
                        prior.mean.device
                    ),
                )
        if "multiple" in self.encoder_mode:
            # determine the number of hidden layers
            if name in self.n_hidden.keys():
                n_hidden = self.n_hidden[name]
            else:
                n_hidden = self.n_hidden["multiple"]
            multi_encoder_kwargs = deepcopy(self.multi_encoder_kwargs)
            multi_encoder_kwargs["n_hidden"] = n_hidden

            # create multiple encoders
            if self.encoder_instance is not None:
                # copy instances
                encoder_ = deepcopy(self.encoder_instance).to(prior.mean.device)
                # convert to pyro module
                to_pyro_module_(encoder_)
                deep_setattr(
                    self,
                    "multiple_encoders." + name,
                    encoder_,
                )
            else:
                # create instances
                deep_setattr(
                    self,
                    "multiple_encoders." + name,
                    self.encoder_class(n_in=self.multiple_n_in, n_out=n_hidden, **multi_encoder_kwargs).to(
                        prior.mean.device
                    ),
                )
        return self.encode(name, prior)

    def _get_params(self, name: str, prior: Distribution):
        if name not in self.amortised_plate_sites["sites"].keys():
            # don't use amortisation unless requested (site in the list)
            return super()._get_params(name, prior)

        args, kwargs = self.args_kwargs  # stored as a tuple of (tuple, dict)
        hidden = self.encode(name, prior)
        try:
            linear_loc = deep_getattr(self.hidden2locs, name)
            bias_loc = deep_getattr(self.bias4locs, name)
            loc = hidden @ linear_loc + bias_loc
            linear_scale = deep_getattr(self.hidden2scales, name)
            bias_scale = deep_getattr(self.bias4scales, name)
            scale = self.softplus((hidden @ linear_scale) + bias_scale - self._init_scale_unconstrained)
            if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
                if self.weight_type == "element-wise":
                    # weight is element-wise
                    linear_weight = deep_getattr(self.hidden2weights, name)
                    bias_weight = deep_getattr(self.bias4weights, name)
                    weight = self.softplus((hidden @ linear_weight) + bias_weight - self._init_weight_unconstrained)
                if self.weight_type == "scalar":
                    # weight is a single value parameter
                    weight = deep_getattr(self.weights, name)
                return loc, scale, weight
            else:
                return loc, scale
        except AttributeError:
            pass

        # Initialize.
        with torch.no_grad():
            init_scale = torch.full((), self._init_scale)
            self._init_scale_unconstrained = self.softplus.inv(init_scale)
            init_weight = torch.full((), self._init_weight)
            self._init_weight_unconstrained = self.softplus.inv(init_weight)

            # determine the number of hidden layers
            if "multiple" in self.encoder_mode:
                if name in self.n_hidden.keys():
                    n_hidden = self.n_hidden[name]
                else:
                    n_hidden = self.n_hidden["multiple"]
            elif "single" in self.encoder_mode:
                n_hidden = self.n_hidden["single"]
            # determine parameter dimensions
            param_dim = (n_hidden, self.amortised_plate_sites["sites"][name])
            bias_dim = (1, self.amortised_plate_sites["sites"][name])
            # generate initial value for linear parameters
            init_param = torch.normal(
                torch.full(size=param_dim, fill_value=0.0, device=prior.mean.device),
                torch.full(
                    size=param_dim, fill_value=(1 * self.init_param_scale) / np.sqrt(n_hidden), device=prior.mean.device
                ),
            )
        deep_setattr(self, "hidden2locs." + name, PyroParam(init_param.clone().detach().requires_grad_(True)))
        deep_setattr(self, "hidden2scales." + name, PyroParam(init_param.clone().detach().requires_grad_(True)))
        deep_setattr(
            self, "bias4locs." + name, PyroParam(torch.full(size=bias_dim, fill_value=0.0, device=prior.mean.device))
        )
        deep_setattr(
            self, "bias4scales." + name, PyroParam(torch.full(size=bias_dim, fill_value=0.0, device=prior.mean.device))
        )
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            if self.weight_type == "scalar":
                # weight is a single value parameter
                deep_setattr(self, "weights." + name, PyroParam(init_weight, constraint=constraints.positive))
            if self.weight_type == "element-wise":
                # weight is element-wise
                deep_setattr(
                    self, "hidden2weights." + name, PyroParam(init_param.clone().detach().requires_grad_(True))
                )
                deep_setattr(
                    self,
                    "bias4weights." + name,
                    PyroParam(torch.full(size=bias_dim, fill_value=0.0, device=prior.mean.device)),
                )
        return self._get_params(name, prior)

    def median(self, *args, **kwargs):
        self._computing_median = True
        try:
            return self(*args, **kwargs)
        finally:
            self._computing_median = False

    def _get_posterior_median(self, name, prior):
        transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
        else:
            loc, scale = self._get_params(name, prior)
        return transform(loc)

    def mutual_information(self, *args, **kwargs):
        self._computing_mi = True
        try:
            return self(*args, **kwargs)
        finally:
            self._computing_mi = False

    def _get_mutual_information(self, name, prior):
        """Approximate the mutual information between x and z
            I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        Returns: Float

        """
        if name not in self.amortised_plate_sites["sites"].keys():
            # if amortisation not used return 0
            return torch.zeros(())

        #### get posterior mean and variance ####
        transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
        else:
            loc, scale = self._get_params(name, prior)

        #### get sample from posterior ####
        z_samples = self.get_posterior(name, prior)

        #### compute mi ####
        x_batch, nz = loc.size()

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+scale.loc()).sum(-1)
        neg_entropy = (-0.5 * nz * torch.log(2 * torch.pi) - 0.5 * (1 + (scale ** 2).log()).sum(-1)).mean()

        # [1, x_batch, nz]
        loc, scale = loc.unsqueeze(0), scale.unsqueeze(0)
        var = scale ** 2

        # (z_batch, x_batch, nz)
        dev = z_samples - loc

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - 0.5 * (
            nz * torch.log(2 * torch.pi) + (scale ** 2).log().sum(-1)
        )

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - torch.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()
