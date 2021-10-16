from contextlib import ExitStack

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.distributions import constraints
from pyro.distributions.util import sum_rightmost
from pyro.infer.autoguide import AutoGuide
from pyro.infer.autoguide.guides import _deep_getattr, _deep_setattr
from pyro.infer.autoguide.initialization import InitMessenger, init_to_feasible
from pyro.infer.autoguide.utils import helpful_support_errors
from pyro.nn.module import PyroModule, PyroParam
from pyro.ops.tensor_utils import periodic_repeat
from torch.distributions import biject_to


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
    :param dict hierarchical_sites: dictionary of sites, the posterior mean of
        which should depend on the value of their parent sites in the model
        hierarchy:
        {
         "site name 1": ["parent site name 1", "parent site name 2", ... ],
         "site name 2": ...
         }
         In case of multi-level hierarhy, parent sites need to be listed
         before child site.
         Initial values for child sites are set to 0 in unconstrained space.
         At the moment, if you need to use informative initialisation you need
         to use independent sites.
    """

    scale_constraint = constraints.softplus_positive

    def __init__(
        self,
        model,
        *,
        init_loc_fn=init_to_feasible,
        init_scale=0.1,
        create_plates=None,
        hierarchical_sites: dict = dict()
    ):
        self.init_loc_fn = init_loc_fn
        self.hierarchical_sites = hierarchical_sites

        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale

        model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self._event_dims = {}
        self.locs = PyroModule()
        self.scales = PyroModule()

        # Initialize guide params
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Collect unconstrained event_dims, which may differ from constrained event_dims.
            with helpful_support_errors(site):
                init_loc = biject_to(site["fn"].support).inv(site["value"].detach()).detach()
            event_dim = site["fn"].event_dim + init_loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim

            # If subsampling, repeat init_value to full size.
            for frame in site["cond_indep_stack"]:
                full_size = getattr(frame, "full_size", frame.size)
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    init_loc = periodic_repeat(init_loc, full_size, dim).contiguous()
            init_scale = torch.full_like(init_loc, self._init_scale)

            # If site has hierarchical parent sites,
            # the hierarchy will determine initial values
            if name in self.hierarchical_sites.keys():
                init_loc = torch.zeros_like(init_loc)

            _deep_setattr(self.locs, name, PyroParam(init_loc, constraints.real, event_dim))
            _deep_setattr(
                self.scales,
                name,
                PyroParam(init_scale, self.scale_constraint, event_dim),
            )

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

        # sample sites without hierarchical dependency
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Don't sample if the site has hierarchical dependency
            if name in self.hierarchical_sites.keys():
                continue
            transform = biject_to(site["fn"].support)

            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])

                site_loc, site_scale = self._get_loc_and_scale(name)
                unconstrained_latent = pyro.sample(
                    name + "_unconstrained",
                    dist.Normal(
                        site_loc,
                        site_scale,
                    ).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True},
                )

                value = transform(unconstrained_latent)
                if poutine.get_mask() is False:
                    log_density = 0.0
                else:
                    log_density = transform.inv.log_abs_det_jacobian(
                        value,
                        unconstrained_latent,
                    )
                    log_density = sum_rightmost(
                        log_density,
                        log_density.dim() - value.dim() + site["fn"].event_dim,
                    )
                delta_dist = dist.Delta(
                    value,
                    log_density=log_density,
                    event_dim=site["fn"].event_dim,
                )

                result[name] = pyro.sample(name, delta_dist)

        # Sample sites with hierarchical dependency
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Don't sample if the site does not have a hierarchical dependency
            if name not in self.hierarchical_sites.keys():
                continue
            transform = biject_to(site["fn"].support)

            # Get the expected value of the site based on hierarchy
            # Get values of parent sites
            parent_names = self.hierarchical_sites[name]
            parent_result = {k: result[k] for k in parent_names}

            # Propagate through a section of the model (block)
            # to get the expected value of the site
            with poutine.block():
                model_block = poutine.block(self.model, expose=[name] + parent_names)
                conditioned = poutine.condition(model_block, data=parent_result)
                conditioned_trace = poutine.trace(conditioned).get_trace(*args, **kwargs)
                site_loc_hierarhical_constrained = conditioned_trace.nodes[name]["value"]

            # transform to unconstrained space
            site_loc_hierarhical_unconstrained = transform.inv(site_loc_hierarhical_constrained)

            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])

                site_loc, site_scale = self._get_loc_and_scale(name)
                # use a combination of hierarchical and independent loc
                unconstrained_latent = pyro.sample(
                    name + "_unconstrained",
                    dist.Normal(
                        site_loc + site_loc_hierarhical_unconstrained,
                        site_scale,
                    ).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True},
                )

                value = transform(unconstrained_latent)
                if poutine.get_mask() is False:
                    log_density = 0.0
                else:
                    log_density = transform.inv.log_abs_det_jacobian(
                        value,
                        unconstrained_latent,
                    )
                    log_density = sum_rightmost(
                        log_density,
                        log_density.dim() - value.dim() + site["fn"].event_dim,
                    )
                delta_dist = dist.Delta(
                    value,
                    log_density=log_density,
                    event_dim=site["fn"].event_dim,
                )

                result[name] = pyro.sample(name, delta_dist)

        return result

    @torch.no_grad()
    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        medians = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # independent locs
            site_loc, _ = self._get_loc_and_scale(name)
            # hierachical component
            if name in self.hierarchical_sites.keys():
                # Get the expected value of the site based on hierarchy
                # Get values of parent sites
                parent_names = self.hierarchical_sites[name]
                parent_support = {k: self.prototype_trace.nodes[k]["fn"].support for k in parent_names}
                parent_medians = {k: biject_to(parent_support[k])(self._get_loc_and_scale(k)[0]) for k in parent_names}

                # Propagate through a section of the model (block)
                # to get the expected value of the site
                with poutine.block():
                    model_block = poutine.block(self.model, expose=[name] + parent_names)
                    conditioned = poutine.condition(model_block, data=parent_medians)
                    conditioned_trace = poutine.trace(conditioned).get_trace(*args, **kwargs)
                    site_loc_hierarhical_constrained = conditioned_trace.nodes[name]["value"]

                # transform to unconstrained space
                site_loc = site_loc + biject_to(site["fn"].support).inv(site_loc_hierarhical_constrained)

            median = biject_to(site["fn"].support)(site_loc)
            if median is site_loc:
                median = median.clone()
            medians[name] = median

        return medians

    @torch.no_grad()
    def quantiles(self, quantiles, *args, **kwargs):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles([0.05, 0.5, 0.95]))

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a tensor of quantile values.
        :rtype: dict
        """
        results = {}

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # independent locs
            site_loc, site_scale = self._get_loc_and_scale(name)
            # hierachical component
            if name in self.hierarchical_sites.keys():
                # Get the expected value of the site based on hierarchy
                # Get values of parent sites
                parent_names = self.hierarchical_sites[name]
                parent_support = {k: self.prototype_trace.nodes[k]["fn"].support for k in parent_names}
                parent_medians = {k: biject_to(parent_support[k])(self._get_loc_and_scale(k)[0]) for k in parent_names}

                # Propagate through a section of the model (block)
                # to get the expected value of the site
                with poutine.block():
                    model_block = poutine.block(self.model, expose=[name] + parent_names)
                    conditioned = poutine.condition(model_block, data=parent_medians)
                    conditioned_trace = poutine.trace(conditioned).get_trace(*args, **kwargs)
                    site_loc_hierarhical_constrained = conditioned_trace.nodes[name]["value"]

                # transform to unconstrained space and add independent component
                site_loc = site_loc + biject_to(site["fn"].support).inv(site_loc_hierarhical_constrained)

            site_quantiles = torch.tensor(quantiles, dtype=site_loc.dtype, device=site_loc.device)
            site_quantiles = site_quantiles.reshape((-1,) + (1,) * site_loc.dim())
            site_quantiles_values = dist.Normal(site_loc, site_scale).icdf(site_quantiles)
            constrained_site_quantiles = biject_to(site["fn"].support)(site_quantiles_values)
            results[name] = constrained_site_quantiles

        return results
