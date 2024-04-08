import gc
import inspect
import logging
from datetime import date
from functools import partial
from typing import Optional, Union

import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import torch
from lightning.pytorch.callbacks import Callback
from pyro import poutine
from pyro.infer.autoguide import AutoNormal, init_to_feasible, init_to_mean
from scipy.sparse import issparse
from scvi import REGISTRY_KEYS
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import parse_device_args
from scvi.module.base import PyroBaseModuleClass
from scvi.train import PyroTrainingPlan as PyroTrainingPlan_scvi
from scvi.utils import track

from ...distributions.AutoAmortisedNormalMessenger import (
    AutoAmortisedHierarchicalNormalMessenger,
    AutoNormalMessenger,
)

logger = logging.getLogger(__name__)


def setup_pyro_model(dataloader, pl_module):
    """Way to warmup Pyro Model and Guide in an automated way.

    Setup occurs before any device movement, so params are iniitalized on CPU.
    """
    for tensors in dataloader:
        tens = {k: t.to(pl_module.device) for k, t in tensors.items()}
        args, kwargs = pl_module.module._get_fn_args_from_batch(tens)
        pl_module.module.guide(*args, **kwargs)
        break


def init_to_value(site=None, values={}, init_fn=init_to_mean):
    if site is None:
        return partial(init_to_value, values=values)
    if site["name"] in values:
        return values[site["name"]]
    else:
        return init_fn(site)


def expand_zeros_along_dim(tensor, size, dim):
    shape = np.array(tensor.shape)
    shape[dim] = size
    return np.zeros(shape)


def complete_tensor_along_dim(tensor, indices, dim, value, mode="put"):
    shape = value.shape
    shape = np.ones(len(shape))
    shape[dim] = len(indices)
    shape = shape.astype(int)
    indices = indices.reshape(shape)
    if mode == "take":
        return np.take_along_axis(arr=tensor, indices=indices, axis=dim)
    np.put_along_axis(arr=tensor, indices=indices, values=value, axis=dim)
    return tensor


def _complete_full_tensors_using_plates(
    means_global,
    means,
    plate_dict,
    obs_plate_sites,
    plate_indices,
    plate_dim,
    named_dims,
):
    # complete full sized tensors with minibatch values given minibatch indices
    for k in means_global.keys():
        # find which and how many plates contain this tensor
        plates = [plate for plate in plate_dict.keys() if k in obs_plate_sites[plate].keys()]
        if len(plates) == 1:
            # if only one plate contains this tensor, complete it using the plate indices
            if k in named_dims.keys():
                dim = named_dims[k]
            else:
                dim = plate_dim[plates[0]]
            means_global[k] = complete_tensor_along_dim(
                means_global[k],
                plate_indices[plates[0]],
                dim,
                means[k],
            )
        elif len(plates) == 2:
            # subset data to index for plate 0 and fill index for plate 1
            if k in named_dims.keys() and (k in obs_plate_sites[list(plate_dict.keys())[0]].keys()):
                dim0 = named_dims[k]
            else:
                dim0 = plate_dim[plates[0]]
            means_global_k = complete_tensor_along_dim(
                means_global[k],
                plate_indices[plates[0]],
                dim0,
                means[k],
                mode="take",
            )
            if k in named_dims.keys() and (k in obs_plate_sites[list(plate_dict.keys())[1]].keys()):
                dim1 = named_dims[k]
            else:
                dim1 = plate_dim[plates[1]]
            means_global_k = complete_tensor_along_dim(
                means_global_k,
                plate_indices[plates[1]],
                dim1,
                means[k],
            )
            # fill index for plate 0 in the full data
            means_global[k] = complete_tensor_along_dim(
                means_global[k],
                plate_indices[plates[0]],
                dim0,
                means_global_k,
            )
            # TODO add a test - observed variables should be identical if this code works correctly
            # This code works correctly but the test needs to be added eventually
            # np.allclose(
            #     samples['data_chromatin'].squeeze(-1).T,
            #     mod_reg.adata_manager.get_from_registry('X')[
            #         :, ~mod_reg.adata_manager.get_from_registry('gene_bool').ravel()
            #     ].toarray()
            # )
        else:
            NotImplementedError(
                f"Posterior sampling/mean/median/quantile not supported for variables with > 2 plates: {k} has {len(plates)}"
            )
    return means_global


class AutoGuideMixinModule:
    """
    This mixin class provides methods for:

    - initialising standard AutoNormalMessenger guides
    - initialising amortised guides (AutoNormalEncoder)
    - initialising amortised guides with special additional inputs

    """

    def _create_autoguide(
        self,
        model,
        amortised,
        encoder_kwargs,
        encoder_mode,
        init_loc_fn=init_to_mean(fallback=init_to_feasible),
        n_cat_list: list = [],
        encoder_instance=None,
        guide_class=AutoNormalMessenger,
        guide_kwargs: Optional[dict] = None,
    ):
        if guide_kwargs is None:
            guide_kwargs = dict()

        if not amortised:
            if getattr(model, "discrete_variables", None) is not None:
                model = poutine.block(model, hide=model.discrete_variables)
            if issubclass(guide_class, poutine.messenger.Messenger):
                # messenger guides don't need create_plates function
                _guide = guide_class(
                    model,
                    init_loc_fn=init_loc_fn,
                    **guide_kwargs,
                )
            else:
                _guide = guide_class(
                    model,
                    init_loc_fn=init_loc_fn,
                    **guide_kwargs,
                    create_plates=self.model.create_plates,
                )
        else:
            encoder_kwargs = encoder_kwargs if isinstance(encoder_kwargs, dict) else dict()
            n_hidden = encoder_kwargs["n_hidden"] if "n_hidden" in encoder_kwargs.keys() else 200
            amortised_vars = model.list_obs_plate_vars()
            if len(amortised_vars["input"]) >= 2:
                encoder_kwargs["n_cat_list"] = n_cat_list
            n_in = amortised_vars["n_in"]
            if getattr(model, "discrete_variables", None) is not None:
                model = poutine.block(model, hide=model.discrete_variables)
            _guide = AutoAmortisedHierarchicalNormalMessenger(
                model,
                amortised_plate_sites=amortised_vars,
                n_in=n_in,
                n_hidden=n_hidden,
                encoder_kwargs=encoder_kwargs,
                encoder_mode=encoder_mode,
                encoder_instance=encoder_instance,
                init_loc_fn=init_loc_fn,
                **guide_kwargs,
            )
        return _guide


class QuantileMixin:
    """
    This mixin class provides methods for:

    - computing median and quantiles of the posterior distribution using both direct and amortised inference

    """

    def _optim_param(
        self,
        lr: float = 0.01,
        autoencoding_lr: float = None,
        clip_norm: float = 200,
        module_names: list = ["encoder", "hidden2locs", "hidden2scales"],
    ):
        # TODO implement custom training method that can use this function.
        # create function which fetches different lr for autoencoding guide
        def optim_param(module_name, param_name):
            # detect variables in autoencoding guide
            if autoencoding_lr is not None and np.any([n in module_name + "." + param_name for n in module_names]):
                return {
                    "lr": autoencoding_lr,
                    # limit the gradient step from becoming too large
                    "clip_norm": clip_norm,
                }
            else:
                return {
                    "lr": lr,
                    # limit the gradient step from becoming too large
                    "clip_norm": clip_norm,
                }

        return optim_param

    def _get_obs_plate_sites_v2(
        self,
        args: list,
        kwargs: dict,
        plate_name: str = None,
        return_observed: bool = False,
        return_deterministic: bool = True,
    ):
        """
        Automatically guess which model sites belong to observation/minibatch plate.
        This function requires minibatch plate name specified in `self.module.list_obs_plate_vars["name"]`.
        Parameters
        ----------
        args
            Arguments to the model.
        kwargs
            Keyword arguments to the model.
        return_observed
            Record samples of observed variables.
        Returns
        -------
        Dictionary with keys corresponding to site names and values to plate dimension.
        """
        if plate_name is None:
            plate_name = self.module.list_obs_plate_vars["name"]

        def try_trace(args, kwargs):
            try:
                trace_ = poutine.trace(self.module.guide).get_trace(*args, **kwargs)
                trace_ = poutine.trace(poutine.replay(self.module.model, trace_)).get_trace(*args, **kwargs)
            except ValueError:
                # if sample is unsuccessful try again
                trace_ = try_trace(args, kwargs)
            return trace_

        trace = try_trace(args, kwargs)

        # find plate dimension
        obs_plate = {
            name: {
                fun.name: fun
                for fun in site["cond_indep_stack"]
                if (fun.name in plate_name) or (fun.name == plate_name)
            }
            for name, site in trace.nodes.items()
            if (
                (site["type"] == "sample")  # sample statement
                and (
                    ((not site.get("is_observed", True)) or return_observed)  # don't save observed unless requested
                    or (site.get("infer", False).get("_deterministic", False) and return_deterministic)
                )  # unless it is deterministic
                and not isinstance(site.get("fn", None), poutine.subsample_messenger._Subsample)  # don't save plates
            )
            if any(f.name == plate_name for f in site["cond_indep_stack"])
        }

        return obs_plate

    def _get_dataloader(
        self,
        batch_size,
        data_loader_indices,
        dl_kwargs={},
    ):
        if dl_kwargs is None:
            dl_kwargs = dict()
        signature_keys = list(inspect.signature(self._data_splitter_cls).parameters.keys())
        if "drop_last" in signature_keys:
            dl_kwargs["drop_last"] = False
        if "shuffle" in signature_keys:
            dl_kwargs["shuffle_training"] = False
        if "shuffle_set_split" in signature_keys:
            dl_kwargs["shuffle_set_split"] = False
        if "indices" in signature_keys:
            dl_kwargs["indices"] = data_loader_indices
        train_dl = self._data_splitter_cls(
            self.adata_manager,
            batch_size=batch_size,
            train_size=1.0,
            **dl_kwargs,
        )
        train_dl.setup()
        train_dl = train_dl.train_dataloader()
        return train_dl

    @torch.inference_mode()
    def _posterior_quantile_minibatch(
        self,
        q: float = 0.5,
        batch_size: int = 2048,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        use_median: bool = True,
        return_observed: bool = False,
        exclude_vars: list = None,
        data_loader_indices=None,
        show_progress: bool = True,
        dl_kwargs: Optional[dict] = None,
    ):
        """
        Compute median of the posterior distribution of each parameter, separating local (minibatch) variable
        and global variables, which is necessary when performing amortised inference.

        Note for developers: requires model class method which lists observation/minibatch plate
        variables (self.module.model.list_obs_plate_vars()).

        Parameters
        ----------
        q
            quantile to compute
        batch_size
            number of observations per batch
        use_gpu
            Bool, use gpu?
        use_median
            Bool, when q=0.5 use median rather than quantile method of the guide

        Returns
        -------
        dictionary {variable_name: posterior quantile}

        """

        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device="torch",
            validate_single_device=True,
        )

        self.module.eval()

        train_dl = self._get_dataloader(
            batch_size=batch_size,
            data_loader_indices=data_loader_indices,
            dl_kwargs=dl_kwargs,
        )

        i = 0
        for tensor_dict in track(
            train_dl,
            style="tqdm",
            description=f"Computing posterior quantile {q}, data batch: ",
            disable=not show_progress,
        ):
            args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
            args = [a.to(device) for a in args]
            kwargs = {k: v.to(device) for k, v in kwargs.items()}
            self.to_device(device)

            if i == 0:
                minibatch_plate_names = self.module.list_obs_plate_vars["name"]
                plates = self.module.model.create_plates(*args, **kwargs)
                if not isinstance(plates, list):
                    plates = [plates]
                # find plate indices & dim
                plate_dict = {
                    plate.name: plate
                    for plate in plates
                    if ((plate.name in minibatch_plate_names) or (plate.name == minibatch_plate_names))
                }
                plate_size = {name: plate.size for name, plate in plate_dict.items()}
                if data_loader_indices is not None:
                    # set total plate size to the number of indices in DL not total number of observations
                    # this option is not really used
                    plate_size = {
                        name: len(train_dl.indices)
                        for name, plate in plate_dict.items()
                        if plate.name == minibatch_plate_names
                    }
                plate_dim = {name: plate.dim for name, plate in plate_dict.items()}
                plate_indices = {name: plate.indices.detach().cpu().numpy() for name, plate in plate_dict.items()}
                # find plate sites
                obs_plate_sites = {
                    plate: self._get_obs_plate_sites_v2(args, kwargs, plate_name=plate, return_observed=return_observed)
                    for plate in plate_dict.keys()
                }
                if use_median and q == 0.5:
                    # use median rather than quantile method
                    def try_median(args, kwargs):
                        try:
                            means_ = self.module.guide.median(*args, **kwargs)
                        except ValueError:
                            # if sample is unsuccessful try again
                            means_ = try_median(args, kwargs)
                        return means_

                    means = try_median(args, kwargs)
                else:

                    def try_quantiles(args, kwargs):
                        try:
                            means_ = self.module.guide.quantiles([q], *args, **kwargs)
                        except ValueError:
                            # if sample is unsuccessful try again
                            means_ = try_quantiles(args, kwargs)
                        return means_

                    means = try_quantiles(args, kwargs)
                valid_sites = self._get_valid_sites(args, kwargs, return_observed=return_observed)
                means = {
                    k: means[k].detach().cpu().numpy()
                    for k in means.keys()
                    if (k not in exclude_vars) and (k in valid_sites)
                }
                means_global = means.copy()
                for plate in plate_dict.keys():
                    # create full sized tensors according to plate size
                    means_global = {
                        k: (
                            expand_zeros_along_dim(
                                means_global[k],
                                plate_size[plate],
                                plate_dim[plate]
                                if not (
                                    (k in getattr(self.module.model, "named_dims", dict()).keys())
                                    and (k in obs_plate_sites[plate].keys())
                                )
                                else self.module.model.named_dims[k],
                            )
                            if k in obs_plate_sites[plate].keys()
                            else means_global[k]
                        )
                        for k in means_global.keys()
                    }
                # complete full sized tensors with minibatch values given minibatch indices
                means_global = _complete_full_tensors_using_plates(
                    means_global=means_global,
                    means=means,
                    plate_dict=plate_dict,
                    obs_plate_sites=obs_plate_sites,
                    plate_indices=plate_indices,
                    plate_dim=plate_dim,
                    named_dims=getattr(self.module.model, "named_dims", dict()),
                )
                if np.all([len(v) == 0 for v in obs_plate_sites.values()]):
                    # if no local variables - don't sample further - return results now
                    break
            else:
                if use_median and q == 0.5:

                    def try_median(args, kwargs):
                        try:
                            means_ = self.module.guide.median(*args, **kwargs)
                        except ValueError:
                            # if sample is unsuccessful try again
                            means_ = try_median(args, kwargs)
                        return means_

                    means = try_median(args, kwargs)
                else:

                    def try_quantiles(args, kwargs):
                        try:
                            means_ = self.module.guide.quantiles([q], *args, **kwargs)
                        except ValueError:
                            # if sample is unsuccessful try again
                            means_ = try_quantiles(args, kwargs)
                        return means_

                    means = try_quantiles(args, kwargs)
                valid_sites = self._get_valid_sites(args, kwargs, return_observed=return_observed)
                means = {
                    k: means[k].detach().cpu().numpy()
                    for k in means.keys()
                    if (k not in exclude_vars) and (k in valid_sites)
                }
                # find plate indices & dim
                plates = self.module.model.create_plates(*args, **kwargs)
                if not isinstance(plates, list):
                    plates = [plates]
                plate_dict = {
                    plate.name: plate
                    for plate in plates
                    if ((plate.name in minibatch_plate_names) or (plate.name == minibatch_plate_names))
                }
                plate_indices = {name: plate.indices.detach().cpu().numpy() for name, plate in plate_dict.items()}
                # TODO - is this correct to call this function again? find plate sites
                obs_plate_sites = {
                    plate: self._get_obs_plate_sites_v2(args, kwargs, plate_name=plate, return_observed=return_observed)
                    for plate in plate_dict.keys()
                }
                # complete full sized tensors with minibatch values given minibatch indices
                means_global = _complete_full_tensors_using_plates(
                    means_global=means_global,
                    means=means,
                    plate_dict=plate_dict,
                    obs_plate_sites=obs_plate_sites,
                    plate_indices=plate_indices,
                    plate_dim=plate_dim,
                    named_dims=getattr(self.module.model, "named_dims", dict()),
                )
            i += 1

        self.module.to(device)

        return means_global

    def posterior_quantile(self, exclude_vars: list = None, batch_size: int = None, **kwargs):
        """
        Compute median of the posterior distribution of each parameter.

        Parameters
        ----------
        q
            Quantile to compute
        use_gpu
            Bool, use gpu?
        use_median
            Bool, when q=0.5 use median rather than quantile method of the guide

        Returns
        -------

        """
        if exclude_vars is None:
            exclude_vars = []
        if kwargs is None:
            kwargs = dict()

        if isinstance(self.module.guide, AutoNormal):
            # median/quantiles in AutoNormal does not require minibatches
            batch_size = None

        if batch_size is None:
            batch_size = self.adata_manager.adata.n_obs
        return self._posterior_quantile_minibatch(exclude_vars=exclude_vars, batch_size=batch_size, **kwargs)


class PltExportMixin:
    r"""
    This mixing class provides methods for common plotting tasks and data export.
    """

    @staticmethod
    def plot_posterior_mu_vs_data(mu, data):
        r"""Plot expected value of the model (e.g. mean of NB distribution) vs observed data

        :param mu: expected value
        :param data: data value
        """

        plt.hist2d(
            np.log10(data.flatten() + 1),
            np.log10(mu.flatten() + 1),
            bins=50,
            norm=matplotlib.colors.LogNorm(),
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("Data, log10")
        plt.ylabel("Posterior expected value, log10")
        plt.title("Reconstruction accuracy")
        plt.tight_layout()

    def plot_history(self, iter_start=0, iter_end=-1, ax=None):
        r"""Plot training history
        Parameters
        ----------
        iter_start
            omit initial iterations from the plot
        iter_end
            omit last iterations from the plot
        ax
            matplotlib axis
        """
        if ax is None:
            ax = plt.gca()
        if iter_end == -1:
            iter_end = len(self.history_["elbo_train"])

        ax.plot(
            np.array(self.history_["elbo_train"].index[iter_start:iter_end]),
            np.array(self.history_["elbo_train"].values.flatten())[iter_start:iter_end],
            label="train",
        )
        ax.legend()
        ax.set_xlim(0, len(self.history_["elbo_train"]))
        ax.set_xlabel("Training epochs")
        ax.set_ylabel("-ELBO loss")
        plt.tight_layout()

    def _export2adata(self, samples):
        r"""
        Export key model variables and samples

        Parameters
        ----------
        samples
            dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``

        Returns
        -------
            Updated dictionary with additional details is saved to ``adata.uns['mod']``.
        """
        # add factor filter and samples of all parameters to unstructured data
        results = {
            "model_name": str(self.module.__class__.__name__),
            "date": str(date.today()),
            "factor_filter": list(getattr(self, "factor_filter", [])),
            "factor_names": list(self.factor_names_),
            "var_names": self.adata.var_names.tolist(),
            "obs_names": self.adata.obs_names.tolist(),
            "post_sample_means": samples["post_sample_means"] if "post_sample_means" in samples else None,
            "post_sample_stds": samples["post_sample_stds"] if "post_sample_stds" in samples else None,
        }
        # add posterior quantiles
        for k, v in samples.items():
            if k.startswith("post_sample_"):
                results[k] = v
        if type(self.factor_names_) is dict:
            results["factor_names"] = self.factor_names_

        return results

    def sample2df_obs(
        self,
        samples: dict,
        site_name: str = "w_sf",
        summary_name: str = "means",
        name_prefix: str = "cell_abundance",
        factor_names_key: str = "",
    ):
        """Export posterior distribution summary for observation-specific parameters
        (e.g. spatial cell abundance) as Pandas data frame
        (means, 5%/95% quantiles or sd of posterior distribution).

        Parameters
        ----------
        samples
            dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``
        site_name
            name of the model parameter to be exported
        summary_name
            posterior distribution summary to return ['means', 'stds', 'q05', 'q95']
        name_prefix
            prefix to add to column names (f'{summary_name}{name_prefix}_{site_name}_{self\.factor_names_}')

        Returns
        -------
        Pandas data frame corresponding to either means, 5%/95% quantiles or sd of the posterior distribution

        """
        if type(self.factor_names_) is dict:
            factor_names_ = self.factor_names_[factor_names_key]
        else:
            factor_names_ = self.factor_names_

        return pd.DataFrame(
            samples[f"post_sample_{summary_name}"].get(site_name, None),
            index=self.adata.obs_names,
            columns=[f"{summary_name}{name_prefix}_{site_name}_{i}" for i in factor_names_],
        )

    def sample2df_vars(
        self,
        samples: dict,
        site_name: str = "gene_factors",
        summary_name: str = "means",
        name_prefix: str = "",
        factor_names_key: str = "",
    ):
        r"""Export posterior distribution summary for variable-specific parameters as Pandas data frame
        (means, 5%/95% quantiles or sd of posterior distribution).

        Parameters
        ----------
        samples
            dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``
        site_name
            name of the model parameter to be exported
        summary_name
            posterior distribution summary to return ('means', 'stds', 'q05', 'q95')
        name_prefix
            prefix to add to column names (f'{summary_name}{name_prefix}_{site_name}_{self\.factor_names_}')

        Returns
        -------
        Pandas data frame corresponding to either means, 5%/95% quantiles or sd of the posterior distribution

        """
        if type(self.factor_names_) is dict:
            factor_names_ = self.factor_names_[factor_names_key]
        else:
            factor_names_ = self.factor_names_
        site = samples[f"post_sample_{summary_name}"].get(site_name, None)
        return pd.DataFrame(
            site,
            columns=self.adata.var_names,
            index=[f"{summary_name}{name_prefix}_{site_name}_{i}" for i in factor_names_],
        ).T

    def plot_QC(self, summary_name: str = "means", use_n_obs: int = 1000):
        """
        Show quality control plots:

        1. Reconstruction accuracy to assess if there are any issues with model training.
           The plot should be roughly diagonal, strong deviations signal problems that need to be investigated.
           Plotting is slow because expected value of mRNA count needs to be computed from model parameters. Random
           observations are used to speed up computation.

        Parameters
        ----------
        summary_name
            posterior distribution summary to use ('means', 'stds', 'q05', 'q95')

        Returns
        -------

        """

        if getattr(self, "samples", False) is False:
            raise RuntimeError("self.samples is missing, please run self.export_posterior() first")
        if use_n_obs is not None:
            ind_x = np.random.choice(
                self.adata_manager.adata.n_obs, np.min((use_n_obs, self.adata.n_obs)), replace=False
            )
        else:
            ind_x = None

        self.expected_nb_param = self.module.model.compute_expected(
            self.samples[f"post_sample_{summary_name}"], self.adata_manager, ind_x=ind_x
        )
        x_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)[ind_x, :]
        if issparse(x_data):
            x_data = np.asarray(x_data.toarray())
        self.plot_posterior_mu_vs_data(self.expected_nb_param["mu"], x_data)


class PyroAggressiveConvergence(Callback):
    """
    A callback to compute/apply aggressive training convergence criteria for amortised inference.
    Motivated by this paper: https://arxiv.org/pdf/1901.05534.pdf
    """

    def __init__(self, dataloader: AnnDataLoader = None, patience: int = 10, tolerance: float = 1e-4) -> None:
        super().__init__()
        self.dataloader = dataloader
        self.patience = patience
        self.tolerance = tolerance

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ) -> None:
        """
        Compute aggressive training convergence criteria for amortised inference.
        """
        pyro_guide = pl_module.module.guide
        if hasattr(pyro_guide, "mutual_information"):
            if self.dataloader is None:
                dl = trainer.datamodule.train_dataloader()
            else:
                dl = self.dataloader
            for tensors in dl:
                tens = {k: t.to(pl_module.device) for k, t in tensors.items()}
                args, kwargs = pl_module.module._get_fn_args_from_batch(tens)
                break
            mi_ = pyro_guide.mutual_information(*args, **kwargs)
            mi_ = np.array([v for v in mi_.values()]).sum()
            pl_module.log("MI", mi_, prog_bar=True)
            if len(pl_module.mi) > 1:
                if abs(mi_ - pl_module.mi[-1]) < self.tolerance:
                    pl_module.n_epochs_patience += 1
            else:
                pl_module.n_epochs_patience = 0
            if pl_module.n_epochs_patience > self.patience:
                # stop aggressive training by setting epoch counter to max epochs
                # pl_module.aggressive_epochs_counter = pl_module.n_aggressive_epochs + 1
                logger.info('Stopped aggressive training after "{}" epochs'.format(pl_module.aggressive_epochs_counter))
            pl_module.mi.append(mi_)


class PyroTrainingPlan(PyroTrainingPlan_scvi):
    def on_train_epoch_end(self):
        """Training epoch end for Pyro training."""
        outputs = self.training_step_outputs
        elbo = 0
        n = 0
        for out in outputs:
            elbo += out["loss"]
            n += 1
        if n > 0:
            elbo /= n
        self.log("elbo_train", elbo, prog_bar=True)
        self.training_step_outputs.clear()
        gc.collect()
        torch.cuda.empty_cache()


class PyroAggressiveTrainingPlan1(PyroTrainingPlan_scvi):
    """
    Lightning module task to train Pyro scvi-tools modules.
    Parameters
    ----------
    pyro_module
        An instance of :class:`~scvi.module.base.PyroBaseModuleClass`. This object
        should have callable `model` and `guide` attributes or methods.
    loss_fn
        A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`.
        If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
    optim
        A Pyro optimizer instance, e.g., :class:`~pyro.optim.Adam`. If `None`,
        defaults to :class:`pyro.optim.Adam` optimizer with a learning rate of `1e-3`.
    optim_kwargs
        Keyword arguments for **default** optimiser :class:`pyro.optim.Adam`.
    n_aggressive_epochs
        Number of epochs in aggressive optimisation of amortised variables.
    n_aggressive_steps
        Number of steps to spend optimising amortised variables before one step optimising global variables.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    """

    def __init__(
        self,
        pyro_module: PyroBaseModuleClass,
        loss_fn: Optional[pyro.infer.ELBO] = None,
        optim: Optional[pyro.optim.PyroOptim] = None,
        optim_kwargs: Optional[dict] = None,
        n_aggressive_epochs: int = 1000,
        n_aggressive_steps: int = 20,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        aggressive_vars: Union[list, None] = None,
        invert_aggressive_selection: bool = False,
    ):
        super().__init__(
            pyro_module=pyro_module,
            loss_fn=loss_fn,
            optim=optim,
            optim_kwargs=optim_kwargs,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
        )

        self.n_aggressive_epochs = n_aggressive_epochs
        self.n_aggressive_steps = n_aggressive_steps
        self.aggressive_steps_counter = 0
        self.aggressive_epochs_counter = 0
        self.mi = []
        self.n_epochs_patience = 0

        # in list not provided use amortised variables for aggressive training
        if aggressive_vars is None:
            aggressive_vars = list(self.module.list_obs_plate_vars["sites"].keys())
            aggressive_vars = aggressive_vars + [f"{i}_initial" for i in aggressive_vars]
            aggressive_vars = aggressive_vars + [f"{i}_unconstrained" for i in aggressive_vars]

        self.aggressive_vars = aggressive_vars
        self.invert_aggressive_selection = invert_aggressive_selection
        # keep frozen variables as frozen
        self.requires_grad_false_vars = [k for k, v in self.module.guide.named_parameters() if not v.requires_grad] + [
            k for k, v in self.module.model.named_parameters() if not v.requires_grad
        ]

        self.svi = pyro.infer.SVI(
            model=pyro_module.model,
            guide=pyro_module.guide,
            optim=self.optim,
            loss=self.loss_fn,
        )

    def change_requires_grad(self, aggressive_vars_status, non_aggressive_vars_status):
        for k, v in self.module.guide.named_parameters():
            if not np.any([i in k for i in self.requires_grad_false_vars]):
                k_in_vars = np.any([i in k for i in self.aggressive_vars])
                # hide variables on the list if they are not hidden
                if k_in_vars and v.requires_grad and (aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables on the list if they are hidden
                if k_in_vars and (not v.requires_grad) and (aggressive_vars_status == "expose"):
                    v.requires_grad = True

                # hide variables not on the list if they are not hidden
                if (not k_in_vars) and v.requires_grad and (non_aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables not on the list if they are hidden
                if (not k_in_vars) and (not v.requires_grad) and (non_aggressive_vars_status == "expose"):
                    v.requires_grad = True

        for k, v in self.module.model.named_parameters():
            if not np.any([i in k for i in self.requires_grad_false_vars]):
                k_in_vars = np.any([i in k for i in self.aggressive_vars])
                # hide variables on the list if they are not hidden
                if k_in_vars and v.requires_grad and (aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables on the list if they are hidden
                if k_in_vars and (not v.requires_grad) and (aggressive_vars_status == "expose"):
                    v.requires_grad = True

                # hide variables not on the list if they are not hidden
                if (not k_in_vars) and v.requires_grad and (non_aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables not on the list if they are hidden
                if (not k_in_vars) and (not v.requires_grad) and (non_aggressive_vars_status == "expose"):
                    v.requires_grad = True

    def on_train_epoch_end(self):
        self.aggressive_epochs_counter += 1

        self.change_requires_grad(
            aggressive_vars_status="expose",
            non_aggressive_vars_status="expose",
        )

        outputs = self.training_step_outputs
        elbo = 0
        n = 0
        for out in outputs:
            elbo += out["loss"]
            n += 1
        if n > 0:
            elbo /= n
        self.log("elbo_train", elbo, prog_bar=True)
        self.training_step_outputs.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        # Set KL weight if necessary.
        # Note: if applied, ELBO loss in progress bar is the effective KL annealed loss, not the true ELBO.
        if self.use_kl_weight:
            kwargs.update({"kl_weight": self.kl_weight})

        if self.aggressive_epochs_counter < self.n_aggressive_epochs:
            if self.aggressive_steps_counter < self.n_aggressive_steps:
                self.aggressive_steps_counter += 1
                # Do parameter update exclusively for amortised variables
                if self.invert_aggressive_selection:
                    self.change_requires_grad(
                        aggressive_vars_status="hide",
                        non_aggressive_vars_status="expose",
                    )
                else:
                    self.change_requires_grad(
                        aggressive_vars_status="expose",
                        non_aggressive_vars_status="hide",
                    )
                loss = torch.Tensor([self.svi.step(*args, **kwargs)])
            else:
                self.aggressive_steps_counter = 0
                # Do parameter update exclusively for non-amortised variables
                if self.invert_aggressive_selection:
                    self.change_requires_grad(
                        aggressive_vars_status="expose",
                        non_aggressive_vars_status="hide",
                    )
                else:
                    self.change_requires_grad(
                        aggressive_vars_status="hide",
                        non_aggressive_vars_status="expose",
                    )
                loss = torch.Tensor([self.svi.step(*args, **kwargs)])
        else:
            # Do parameter update for both types of variables
            self.change_requires_grad(
                aggressive_vars_status="expose",
                non_aggressive_vars_status="expose",
            )
            loss = torch.Tensor([self.svi.step(*args, **kwargs)])

        return {"loss": loss}


class PyroAggressiveTrainingPlan(PyroAggressiveTrainingPlan1):
    """
    Lightning module task to train Pyro scvi-tools modules.
    Parameters
    ----------
    pyro_module
        An instance of :class:`~scvi.module.base.PyroBaseModuleClass`. This object
        should have callable `model` and `guide` attributes or methods.
    loss_fn
        A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`.
        If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
    optim
        A Pyro optimizer instance, e.g., :class:`~pyro.optim.Adam`. If `None`,
        defaults to :class:`pyro.optim.Adam` optimizer with a learning rate of `1e-3`.
    optim_kwargs
        Keyword arguments for **default** optimiser :class:`pyro.optim.Adam`.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    """

    def __init__(
        self,
        scale_elbo: Union[float, None] = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if scale_elbo != 1.0:
            self.svi = pyro.infer.SVI(
                model=poutine.scale(self.module.model, scale_elbo),
                guide=poutine.scale(self.module.guide, scale_elbo),
                optim=self.optim,
                loss=self.loss_fn,
            )
        else:
            self.svi = pyro.infer.SVI(
                model=self.module.model,
                guide=self.module.guide,
                optim=self.optim,
                loss=self.loss_fn,
            )
