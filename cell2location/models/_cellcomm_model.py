import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from pyro import clear_param_store
from pyro.infer import Trace_ELBO, TraceEnum_ELBO
from pyro.nn import PyroModule
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from scvi.dataloaders import DeviceBackedDataSplitter
from scvi.model.base import BaseModelClass, PyroSampleMixin, PyroSviTrainMixin
from scvi.train import PyroTrainingPlan
from scvi.utils import setup_anndata_dsp

from cell2location.dataloaders._defined_grid_dataloader import SpatialGridDataSplitter
from cell2location.models._cellcomm_module import CellCommModule
from cell2location.models.base._pyro_base_loc_module import Cell2locationBaseModule
from cell2location.models.base._pyro_mixin import (
    PltExportMixin,
    QuantileMixin,
    setup_pyro_model,
)


class CellCommModel(QuantileMixin, PyroSampleMixin, PyroSviTrainMixin, PltExportMixin, BaseModelClass):
    r"""
    Cell2location model. User-end model class. See Module class for description of the model (incl. math).

    Parameters
    ----------
    adata
        spatial AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    cell_state_df
        pd.DataFrame with reference expression signatures for each gene (rows) in each cell type/population (columns).
    **model_kwargs
        Keyword args for :class:`~cell2location.models.LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel`

    Examples
    --------
    TODO add example
    >>>
    """

    def __init__(
        self,
        adata: AnnData,
        receptor_abundance_df: pd.DataFrame,
        model_class: Optional[PyroModule] = None,
        on_load_batch_size: Optional[int] = None,
        **model_kwargs,
    ):
        # in case any other model was created before that shares the same parameter names.
        clear_param_store()

        super().__init__(adata)

        self.mi_ = []

        if model_class is None:
            model_class = CellCommModule

        self.receptor_abundance_ = receptor_abundance_df
        self.n_factors_ = receptor_abundance_df.shape[1]
        self.factor_names_ = receptor_abundance_df.columns.values

        if "tiles" in self.adata_manager.data_registry:
            on_load_batch_size = 1
            self._data_splitter_cls = SpatialGridDataSplitter
            logging.info("Updating data splitter to SpatialGridDataSplitter.")
        self.module = Cell2locationBaseModule(
            model=model_class,
            n_obs=self.summary_stats["n_cells"],
            n_vars=self.summary_stats["n_vars"],
            n_factors=self.n_factors_,
            n_batch=self.summary_stats["n_batch"],
            receptor_abundance=self.receptor_abundance_.values.astype("float32"),
            on_load_kwargs={
                "batch_size": on_load_batch_size,
                "max_epochs": 1,
            },
            **model_kwargs,
        )
        self._model_summary_string = f'CellComm model with the following params: \nn_labels: {self.n_factors_} \nn_batch: {self.summary_stats["n_batch"]} '
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        signal_abundance_key: Optional[str] = None,
        cell_abundance_key: Optional[str] = None,
        cell_abundance_lvl2_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        position_key: Optional[str] = None,
        tiles_key: Optional[str] = None,
        tiles_unexpanded_key: Optional[str] = None,
        in_tissue_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs).astype("int64")
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        if signal_abundance_key is not None:
            anndata_fields.append(ObsmField("signal_abundance", signal_abundance_key))
        if cell_abundance_key is not None:
            anndata_fields.append(ObsmField("w_sf", cell_abundance_key))
        if cell_abundance_lvl2_key is not None:
            anndata_fields.append(ObsmField("w_sf_lvl2", cell_abundance_lvl2_key))
        if position_key is not None:
            anndata_fields.append(ObsmField("positions", position_key))
        if tiles_key is not None:
            anndata_fields.append(ObsmField("tiles", tiles_key))
        if tiles_unexpanded_key is not None:
            anndata_fields.append(ObsmField("tiles_unexpanded", tiles_unexpanded_key))
        if in_tissue_key is not None:
            anndata_fields.append(NumericalObsField("in_tissue", in_tissue_key))

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int = 30000,
        batch_size: int = None,
        train_size: float = 1,
        lr: float = 0.002,
        num_particles: int = 1,
        scale_elbo: float = "auto",
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        early_stopping: bool = False,
        training_plan: Optional[PyroTrainingPlan] = None,
        plan_kwargs: Optional[dict] = None,
        datasplitter_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        %(param_use_gpu)s
        %(param_accelerator)s
        %(param_device)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set are split in the
            sequential order of the data according to `validation_size` and `train_size` percentages.
        batch_size
            Minibatch size to use during training. If `None`, no minibatching occurs and all
            data is copied to device (e.g., GPU).
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        lr
            Optimiser learning rate (default optimiser is :class:`~pyro.optim.ClippedAdam`).
            Specifying optimiser via plan_kwargs overrides this choice of lr.
        training_plan
            Training plan :class:`~scvi.train.PyroTrainingPlan`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.PyroTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        # if max_epochs is None:
        #    max_epochs = get_max_epochs_heuristic(self.adata.n_obs, epochs_cap=1000)
        if datasplitter_kwargs is None:
            datasplitter_kwargs = dict()

        if issubclass(self._data_splitter_cls, SpatialGridDataSplitter):
            self.module.model.n_tiles = batch_size

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
        if lr is not None and "optim" not in plan_kwargs.keys():
            plan_kwargs.update({"optim_kwargs": {"lr": lr}})
        if getattr(self.module.model, "discrete_variables", None) and (len(self.module.model.discrete_variables) > 0):
            plan_kwargs["loss_fn"] = TraceEnum_ELBO(num_particles=num_particles)
        else:
            plan_kwargs["loss_fn"] = Trace_ELBO(num_particles=num_particles)
        if scale_elbo != 1.0:
            if scale_elbo == "auto":
                scale_elbo = 1.0 / (self.summary_stats["n_cells"] * self.summary_stats["n_vars"])
            plan_kwargs["scale_elbo"] = scale_elbo

        if batch_size is None:
            # use data splitter which moves data to GPU once
            data_splitter = DeviceBackedDataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                accelerator=accelerator,
                device=device,
            )
        else:
            data_splitter = self._data_splitter_cls(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                shuffle_set_split=shuffle_set_split,
                batch_size=batch_size,
                **datasplitter_kwargs,
            )

        if training_plan is None:
            training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]

        if "callbacks" not in trainer_kwargs.keys():
            trainer_kwargs["callbacks"] = []

        # Initialise pyro model with data
        from copy import copy

        dl = copy(data_splitter)
        dl.setup()
        dl = dl.train_dataloader()
        setup_pyro_model(dl, training_plan)

        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=device,
            **trainer_kwargs,
        )
        return runner()

    def export_posterior(
        self,
        adata,
        sample_kwargs: Optional[dict] = None,
        export_slot: str = "mod",
        add_to_obsm: list = ["means", "stds", "q05", "q95"],
        use_quantiles: bool = False,
    ):
        """
        Summarise posterior distribution and export results (cell abundance) to anndata object:

        1. adata.obsm: Estimated cell abundance as pd.DataFrames for each posterior distribution summary `add_to_obsm`,
            posterior mean, sd, 5% and 95% quantiles (['means', 'stds', 'q05', 'q95']).
            If export to adata.obsm fails with error, results are saved to adata.obs instead.
        2. adata.uns: Posterior of all parameters, model name, date,
            cell type names ('factor_names'), obs and var names.

        Parameters
        ----------
        adata
            anndata object where results should be saved
        sample_kwargs
            arguments for self.sample_posterior (generating and summarising posterior samples), namely:
                num_samples - number of samples to use (Default = 1000).
                batch_size - data batch size (keep low enough to fit on GPU, default 2048).
                use_gpu - use gpu for generating samples?
        export_slot
            adata.uns slot where to export results
        add_to_obsm
            posterior distribution summary to export in adata.obsm (['means', 'stds', 'q05', 'q95']).
        use_quantiles
            compute quantiles directly (True, more memory efficient) or use samples (False, default).
            If True, means and stds cannot be computed so are not exported and returned.
        Returns
        -------

        """

        sample_kwargs = sample_kwargs if isinstance(sample_kwargs, dict) else dict()

        # get posterior distribution summary
        if use_quantiles:
            add_to_obsm = [i for i in add_to_obsm if (i not in ["means", "stds"]) and ("q" in i)]
            if len(add_to_obsm) == 0:
                raise ValueError("No quantiles to export - please add add_to_obsm=['q05', 'q50', 'q95'].")
            self.samples = dict()
            for i in add_to_obsm:
                q = float(f"0.{i[1:]}")
                self.samples[f"post_sample_{i}"] = self.posterior_quantile(q=q, **sample_kwargs)
        else:
            # generate samples from posterior distributions for all parameters
            # and compute mean, 5%/95% quantiles and standard deviation
            self.samples = self.sample_posterior(**sample_kwargs)

        # export posterior distribution summary for all parameters and
        # annotation (model, date, var, obs and cell type names) to anndata object
        adata.uns[export_slot] = self._export2adata(self.samples)

        # add estimated cell abundance as dataframe to obsm in anndata
        # first convert np.arrays to pd.DataFrames with cell type and observation names
        # data frames contain mean, 5%/95% quantiles and standard deviation, denoted by a prefix
        for k in add_to_obsm:
            sample_df = self.sample2df_obs(
                self.samples,
                site_name="w_sf_cell_comm",
                summary_name=k,
                name_prefix="predicted_cell_abundance",
            )
            try:
                adata.obsm[f"{k}_cell_abundance_w_sf"] = sample_df.loc[adata.obs.index, :]
            except ValueError:
                # Catching weird error with obsm: `ValueError: value.index does not match parent’s axis 1 names`
                adata.obs[sample_df.columns] = sample_df.loc[adata.obs.index, :]

        return adata
