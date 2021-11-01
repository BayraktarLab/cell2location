from typing import Callable, Dict, Optional, Union

import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.distributions.distribution import Distribution

# from pyro.distributions.transforms import SoftplusTransform
from pyro.infer.autoguide import AutoHierarchicalNormalMessenger
from pyro.infer.autoguide.initialization import init_to_feasible, init_to_mean
from pyro.infer.autoguide.utils import (
    deep_getattr,
    deep_setattr,
    helpful_support_errors,
)

# from pyro.infer.effect_elbo import GuideMessenger
from pyro.nn.module import PyroModule, PyroParam  # , pyro_method, to_pyro_module_

# from pyro.poutine.runtime import get_plates
from scvi._compat import Literal
from scvi.nn import FCLayers
from torch.distributions import biject_to, constraints


class FCLayersPyro(FCLayers, PyroModule):
    pass


class AutoAmortisedNormalMessengerMixin:
    """
    EXPERIMENTAL Automatic :class:`~pyro.infer.effect_elbo.GuideMessenger` ,
    intended for use with :class:`~pyro.infer.effect_elbo.Effect_ELBO` or
    similar.

    The mean-field posterior at any site is a transformed normal distribution,
    the mean of which depends on the value of that site given its dependencies in the model:

        loc = loc + transform.inv(prior.mean) * weight

    Where the value of `prior.mean` is conditional on upstream sites in the model.
    This approach doesn't work for distributions that don't have the mean.

    Derived classes may override particular sites and use this simply as a
    default, see AutoNormalMessenger documentation for example.

    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    :param float init_weight: Initial value for the weight of the contribution
        of hierarchical sites to posterior mean for each latent variable.
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
        init_loc_fn: Callable = init_to_mean(fallback=init_to_feasible),
        init_scale: float = 0.1,
        init_weight: float = 1.0,
        hierarchical_sites: Optional[list] = None,
    ):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        super().__init__(model)
        self.init_loc_fn = init_loc_fn
        self._init_scale = init_scale
        self._init_weight = init_weight
        self._hierarchical_sites = hierarchical_sites
        self._computing_median = False

    def get_posterior(
        self,
        name: str,
        prior: Distribution,
        upstream_values: Dict[str, torch.Tensor],
    ) -> Union[Distribution, torch.Tensor]:
        if self._computing_median:
            return self._get_posterior_median(name, prior)

        with helpful_support_errors({"name": name, "fn": prior}):
            transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            # If hierarchical_sites not specified all sites are assumed to be hierarchical
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
            posterior = dist.TransformedDistribution(
                dist.Normal(loc, scale).to_event(transform.domain.event_dim),
                transform.with_cache(),
            )
            return posterior
        else:
            # Fall back to mean field when hierarchical_sites list is not empty and site not in the list.
            return super().get_posterior(name, prior, upstream_values)

    def _get_params(self, name: str, prior: Distribution):
        try:
            loc = deep_getattr(self.locs, name)
            scale = deep_getattr(self.scales, name)
            if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
                weight = deep_getattr(self.weights, name)
                return loc, scale, weight
            else:
                return loc, scale
        except AttributeError:
            pass

        # Initialize.
        with poutine.block(), torch.no_grad():
            transform = biject_to(prior.support)
            event_dim = transform.domain.event_dim
            constrained = self.init_loc_fn({"name": name, "fn": prior}).detach()
            unconstrained = transform.inv(constrained)
            init_loc = self._remove_outer_plates(unconstrained, event_dim)
            init_scale = torch.full_like(init_loc, self._init_scale)
            if self.weight_type == "scalar":
                # weight is a single value parameter
                init_weight = torch.full((), self._init_weight)
            if self.weight_type == "element-wise":
                # weight is element-wise
                init_weight = torch.full_like(init_loc, self._init_weight)
            # if site is hierarchical substract contribution of dependencies from init_loc
            if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
                init_loc = init_loc - init_weight * transform.inv(prior.mean)

        deep_setattr(self, "locs." + name, PyroParam(init_loc, event_dim=event_dim))
        deep_setattr(
            self,
            "scales." + name,
            PyroParam(init_scale, constraint=constraints.positive, event_dim=event_dim),
        )
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            if self.weight_type == "scalar":
                # weight is a single value parameter
                deep_setattr(
                    self,
                    "weights." + name,
                    PyroParam(init_weight, constraint=constraints.positive),
                )
            if self.weight_type == "element-wise":
                # weight is element-wise
                deep_setattr(
                    self,
                    "weights." + name,
                    PyroParam(
                        init_weight,
                        constraint=constraints.positive,
                        event_dim=event_dim,
                    ),
                )

        return self._get_params(name, prior)

    def _get_posterior_median(self, name, prior):
        transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
        else:
            loc, scale = self._get_params(name, prior)
        return transform(loc)


class AutoAmortisedHierarchicalNormalMessenger(AutoHierarchicalNormalMessenger, AutoAmortisedNormalMessengerMixin):
    """
    EXPERIMENTAL Automatic :class:`~pyro.infer.effect_elbo.GuideMessenger` ,
    intended for use with :class:`~pyro.infer.effect_elbo.Effect_ELBO` or
    similar.

    The mean-field posterior at any site is a transformed normal distribution,
    the mean of which depends on the value of that site given its dependencies in the model:

        loc = loc + transform.inv(prior.mean) * weight

    Where the value of `prior.mean` is conditional on upstream sites in the model.
    This approach doesn't work for distributions that don't have the mean.

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
    :param float init_weight: Initial value for the weight of the contribution
        of hierarchical sites to posterior mean for each latent variable.
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
        init_param=0,
        init_param_scale: float = 1 / 50,
        scales_offset: float = -2,
        encoder_class=FCLayersPyro,
        encoder_kwargs=None,
        multi_encoder_kwargs=None,
        encoder_instance: torch.nn.Module = None,
        create_plates=None,
        encoder_mode: Literal["single", "multiple", "single-multiple"] = "single",
        hierarchical_sites: Optional[list] = None,
    ):
        # if not isinstance(init_scale, float) or not (init_scale > 0):
        #    raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        super().__init__(model)
        # self.init_loc_fn = init_loc_fn
        # self._init_scale = init_scale
        # self._init_weight = init_weight
        self._hierarchical_sites = hierarchical_sites
        self._computing_median = False

    def get_posterior(
        self,
        name: str,
        prior: Distribution,
        upstream_values: Dict[str, torch.Tensor],
    ) -> Union[Distribution, torch.Tensor]:
        if self._computing_median:
            return self._get_posterior_median(name, prior)

        with helpful_support_errors({"name": name, "fn": prior}):
            transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            # If hierarchical_sites not specified all sites are assumed to be hierarchical
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
            posterior = dist.TransformedDistribution(
                dist.Normal(loc, scale).to_event(transform.domain.event_dim),
                transform.with_cache(),
            )
            return posterior
        else:
            # Fall back to mean field when hierarchical_sites list is not empty and site not in the list.
            return super().get_posterior(name, prior, upstream_values)

    def _get_params(self, name: str, prior: Distribution):
        try:
            loc = deep_getattr(self.locs, name)
            scale = deep_getattr(self.scales, name)
            if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
                weight = deep_getattr(self.weights, name)
                return loc, scale, weight
            else:
                return loc, scale
        except AttributeError:
            pass

        # Initialize.
        with poutine.block(), torch.no_grad():
            transform = biject_to(prior.support)
            event_dim = transform.domain.event_dim
            constrained = self.init_loc_fn({"name": name, "fn": prior}).detach()
            unconstrained = transform.inv(constrained)
            init_loc = self._remove_outer_plates(unconstrained, event_dim)
            init_scale = torch.full_like(init_loc, self._init_scale)
            if self.weight_type == "scalar":
                # weight is a single value parameter
                init_weight = torch.full((), self._init_weight)
            if self.weight_type == "element-wise":
                # weight is element-wise
                init_weight = torch.full_like(init_loc, self._init_weight)
            # if site is hierarchical substract contribution of dependencies from init_loc
            if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
                init_loc = init_loc - init_weight * transform.inv(prior.mean)

        deep_setattr(self, "locs." + name, PyroParam(init_loc, event_dim=event_dim))
        deep_setattr(
            self,
            "scales." + name,
            PyroParam(init_scale, constraint=constraints.positive, event_dim=event_dim),
        )
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            if self.weight_type == "scalar":
                # weight is a single value parameter
                deep_setattr(
                    self,
                    "weights." + name,
                    PyroParam(init_weight, constraint=constraints.positive),
                )
            if self.weight_type == "element-wise":
                # weight is element-wise
                deep_setattr(
                    self,
                    "weights." + name,
                    PyroParam(
                        init_weight,
                        constraint=constraints.positive,
                        event_dim=event_dim,
                    ),
                )

        return self._get_params(name, prior)

    def _get_posterior_median(self, name, prior):
        transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
        else:
            loc, scale = self._get_params(name, prior)
        return transform(loc)
