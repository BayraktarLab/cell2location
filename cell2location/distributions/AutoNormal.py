# This code is adapted from Pyro:
# 1. softplus transformation for Normal Mean Field approximation
#     unconstrained -> softplus -> sigma
# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import operator
import warnings
import weakref
from contextlib import ExitStack  # python 3

import torch
from torch import nn
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
from pyro.distributions.util import sum_rightmost
from pyro.infer.autoguide.initialization import InitMessenger, init_to_mean
from pyro.nn import PyroModule, PyroParam
from pyro.ops.tensor_utils import periodic_repeat
from pyro.infer.autoguide.guides import AutoGuide

def _deep_setattr(obj, key, val):
    """
    Set an attribute `key` on the object. If any of the prefix attributes do
    not exist, they are set to :class:`~pyro.nn.PyroModule`.
    """

    def _getattr(obj, attr):
        obj_next = getattr(obj, attr, None)
        if obj_next is not None:
            return obj_next
        setattr(obj, attr, PyroModule())
        return getattr(obj, attr)

    lpart, _, rpart = key.rpartition(".")
    # Recursive getattr while setting any prefix attributes to PyroModule
    if lpart:
        obj = functools.reduce(_getattr, [obj] + lpart.split('.'))
    setattr(obj, rpart, val)

def _deep_getattr(obj, key):
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


class AutoNormal(AutoGuide):
    """This implementation of :class:`AutoGuide` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    It should be equivalent to :class: `AutoDiagonalNormal` , but with
    more convenient site names and with better support for
    :class:`~pyro.infer.trace_mean_field_elbo.TraceMeanField_ELBO` .

    In :class:`AutoDiagonalNormal` , if your model has N named
    parameters with dimensions k_i and sum k_i = D, you get a single
    vector of length D for your mean, and a single vector of length D
    for sigmas.  This guide gives you N distinct normals that you can
    call by name.

    Usage::

        guide = AutoNormal(model)
        svi = SVI(model, guide, ...)

    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`pyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """
    def __init__(self, model, *,
                 init_loc_fn=init_to_mean,
                 init_scale=0.0,
                 create_plates=None):
        self.init_loc_fn = init_loc_fn

        if not isinstance(init_scale, float): # or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale

        model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

        self.softplus = nn.Softplus()

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self._event_dims = {}
        self._cond_indep_stacks = {}
        self.locs = PyroModule()
        self.scales = PyroModule()

        # Initialize guide params
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Collect unconstrained event_dims, which may differ from constrained event_dims.
            init_loc = biject_to(site["fn"].support).inv(site["value"].detach()).detach()
            event_dim = site["fn"].event_dim + init_loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim

            # Collect independence contexts.
            self._cond_indep_stacks[name] = site["cond_indep_stack"]

            # If subsampling, repeat init_value to full size.
            for frame in site["cond_indep_stack"]:
                full_size = getattr(frame, "full_size", frame.size)
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    init_loc = periodic_repeat(init_loc, full_size, dim).contiguous()
            init_scale = torch.full_like(init_loc, self._init_scale)

            _deep_setattr(self.locs, name, PyroParam(init_loc, constraints.real, event_dim))
            _deep_setattr(self.scales, name, PyroParam(init_scale, constraints.real, event_dim))

    def _get_loc_and_scale(self, name):
        site_loc = _deep_getattr(self.locs, name)
        site_scale = _deep_getattr(self.scales, name)
        return site_loc, site_scale

    def forward(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        .. note:: This method is used internally by :class:`~torch.nn.Module`.
            Users should instead use :meth:`~torch.nn.Module.__call__`.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            transform = biject_to(site["fn"].support)

            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])

                site_loc, site_scale = self._get_loc_and_scale(name)
                unconstrained_latent = pyro.sample(
                    name + "_unconstrained",
                    dist.Normal(
                        site_loc, self.softplus(site_scale),
                    ).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True}
                )

                value = transform(unconstrained_latent)
                log_density = transform.inv.log_abs_det_jacobian(value, unconstrained_latent)
                log_density = sum_rightmost(log_density, log_density.dim() - value.dim() + site["fn"].event_dim)
                delta_dist = dist.Delta(value, log_density=log_density,
                                        event_dim=site["fn"].event_dim)

                result[name] = pyro.sample(name, delta_dist)

        return result


    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        medians = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            site_loc, _ = self._get_loc_and_scale(name)
            median = biject_to(site["fn"].support)(site_loc)
            if median is site_loc:
                median = median.clone()
            medians[name] = median

        return medians


    def quantiles(self, quantiles, *args, **kwargs):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles([0.05, 0.5, 0.95]))

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        results = {}

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            site_loc, site_scale = self._get_loc_and_scale(name)

            site_quantiles = torch.tensor(quantiles, dtype=site_loc.dtype, device=site_loc.device)
            site_quantiles_values = dist.Normal(site_loc, self.softplus(site_scale)).icdf(site_quantiles)
            constrained_site_quantiles = biject_to(site["fn"].support)(site_quantiles_values)
            results[name] = constrained_site_quantiles

        return results
