import numpy as np
import torch
from pyro import poutine
from scvi.data import synthetic_iid
from scvi.dataloaders import AnnDataLoader

from cell2location import run_colocation
from cell2location.models import Cell2location, RegressionModel
from cell2location.models.simplified._cell2location_v3_no_factorisation_module import (
    LocationModelMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
)
from cell2location.models.simplified._cell2location_v3_no_mg_module import (
    LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelNoMGPyroModel,
)


def test_cell2location():
    save_path = "./cell2location_model_test"
    if torch.cuda.is_available():
        use_gpu = int(torch.cuda.is_available())
    else:
        use_gpu = False
    dataset = synthetic_iid(n_labels=5)
    RegressionModel.setup_anndata(dataset, labels_key="labels", batch_key="batch")

    # train regression model to get signatures of cell types
    sc_model = RegressionModel(dataset)
    # test full data training
    sc_model.train(max_epochs=1, use_gpu=use_gpu)
    # test minibatch training
    sc_model.train(max_epochs=1, batch_size=1000, use_gpu=use_gpu)
    # export the estimated cell abundance (summary of the posterior distribution)
    dataset = sc_model.export_posterior(dataset, sample_kwargs={"num_samples": 10})
    # test plot_QC
    sc_model.plot_QC()
    # test quantile export
    dataset = sc_model.export_posterior(dataset, use_quantiles=True)
    sc_model.plot_QC(summary_name="q05")
    # test save/load
    sc_model.save(save_path, overwrite=True, save_anndata=True)
    sc_model = RegressionModel.load(save_path)
    # export estimated expression in each cluster
    if "means_per_cluster_mu_fg" in dataset.varm.keys():
        inf_aver = dataset.varm["means_per_cluster_mu_fg"][
            [f"means_per_cluster_mu_fg_{i}" for i in dataset.uns["mod"]["factor_names"]]
        ].copy()
    else:
        inf_aver = dataset.var[[f"means_per_cluster_mu_fg_{i}" for i in dataset.uns["mod"]["factor_names"]]].copy()
    inf_aver.columns = dataset.uns["mod"]["factor_names"]

    ### test default cell2location model ###
    Cell2location.setup_anndata(dataset, batch_key="batch")
    ##  full data  ##
    st_model = Cell2location(dataset, cell_state_df=inf_aver, N_cells_per_location=30, detection_alpha=200)
    # test full data training
    st_model.train(max_epochs=1, use_gpu=use_gpu)
    # export the estimated cell abundance (summary of the posterior distribution)
    # full data
    dataset = st_model.export_posterior(dataset, sample_kwargs={"num_samples": 10, "batch_size": st_model.adata.n_obs})
    # test quantile export
    dataset = st_model.export_posterior(dataset, use_quantiles=True)
    st_model.plot_QC(summary_name="q05")
    ##  minibatches of locations  ##
    Cell2location.setup_anndata(dataset, batch_key="batch")
    st_model = Cell2location(dataset, cell_state_df=inf_aver, N_cells_per_location=30, detection_alpha=200)
    # test minibatch training
    st_model.train(max_epochs=1, batch_size=50, use_gpu=use_gpu)
    # export the estimated cell abundance (summary of the posterior distribution)
    # minibatches of locations
    dataset = st_model.export_posterior(dataset, sample_kwargs={"num_samples": 10, "batch_size": 50})
    # test plot_QC
    st_model.plot_QC()
    # test save/load
    st_model.save(save_path, overwrite=True, save_anndata=True)
    st_model = Cell2location.load(save_path)
    # export the estimated cell abundance (summary of the posterior distribution)
    # minibatches of locations
    dataset = st_model.export_posterior(dataset, sample_kwargs={"num_samples": 10, "batch_size": 50})
    # test computing any quantile of the posterior distribution
    if not isinstance(st_model.module.guide, poutine.messenger.Messenger):
        st_model.posterior_quantile(q=0.5, use_gpu=use_gpu)
    # test computing median
    if True:
        if use_gpu:
            device = f"cuda:{use_gpu}"
        else:
            device = "cpu"
        train_dl = AnnDataLoader(st_model.adata_manager, shuffle=False, batch_size=50)
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            args, kwargs = st_model.module._get_fn_args_from_batch(batch)
            break
        st_model.module.guide.median(*args, **kwargs)
    # test computing expected expression per cell type
    st_model.module.model.compute_expected_per_cell_type(st_model.samples["post_sample_q05"], st_model.adata_manager)
    ### test amortised inference with default cell2location model ###
    ##  full data  ##
    Cell2location.setup_anndata(dataset, batch_key="batch")
    st_model = Cell2location(
        dataset,
        cell_state_df=inf_aver,
        N_cells_per_location=30,
        detection_alpha=200,
        amortised=True,
        encoder_mode="multiple",
    )
    # test minibatch training
    st_model.train(max_epochs=1, batch_size=20, use_gpu=use_gpu)
    st_model.train_aggressive(
        max_epochs=3, batch_size=20, plan_kwargs={"n_aggressive_epochs": 1, "n_aggressive_steps": 5}, use_gpu=use_gpu
    )
    # test computing median
    if True:
        if use_gpu:
            device = f"cuda:{use_gpu}"
        else:
            device = "cpu"
        train_dl = AnnDataLoader(st_model.adata_manager, shuffle=False, batch_size=50)
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            args, kwargs = st_model.module._get_fn_args_from_batch(batch)
            break
        st_model.module.guide.median(*args, **kwargs)
        st_model.module.guide.quantiles([0.5], *args, **kwargs)
        st_model.module.guide.mutual_information(*args, **kwargs)
    # export the estimated cell abundance (summary of the posterior distribution)
    # minibatches of locations
    dataset = st_model.export_posterior(dataset, sample_kwargs={"num_samples": 10, "batch_size": 50})

    ### test downstream analysis ###
    _, _ = run_colocation(
        dataset,
        model_name="CoLocatedGroupsSklearnNMF",
        train_args={
            "n_fact": np.arange(3, 4),  # IMPORTANT: use a wider range of the number of factors (5-30)
            "sample_name_col": "batch",  # columns in adata_vis.obs that identifies sample
            "n_restarts": 2,  # number of training restarts
        },
        export_args={"path": f"{save_path}/CoLocatedComb/"},
    )

    ### test simplified cell2location models ###
    ##  no m_g  ##
    Cell2location.setup_anndata(dataset, batch_key="batch")
    st_model = Cell2location(
        dataset,
        cell_state_df=inf_aver,
        N_cells_per_location=30,
        detection_alpha=200,
        model_class=LocationModelMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
    )
    # test full data training
    st_model.train(max_epochs=1, use_gpu=use_gpu)
    # export the estimated cell abundance (summary of the posterior distribution)
    # full data
    dataset = st_model.export_posterior(dataset, sample_kwargs={"num_samples": 10, "batch_size": st_model.adata.n_obs})
    ##  no w_sf factorisation  ##
    Cell2location.setup_anndata(dataset, batch_key="batch")
    st_model = Cell2location(
        dataset,
        cell_state_df=inf_aver,
        N_cells_per_location=30,
        detection_alpha=200,
        model_class=LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelNoMGPyroModel,
    )
    # test full data training
    st_model.train(max_epochs=1, use_gpu=use_gpu)
    # export the estimated cell abundance (summary of the posterior distribution)
    # full data
    st_model.export_posterior(dataset, sample_kwargs={"num_samples": 10, "batch_size": st_model.adata.n_obs})
