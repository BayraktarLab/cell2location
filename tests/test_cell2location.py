import numpy as np
import torch
from pyro.infer.autoguide import AutoHierarchicalNormalMessenger
from scvi.data import synthetic_iid
from scvi.dataloaders import AnnDataLoader

from cell2location import compute_weighted_average_around_target, run_colocation
from cell2location.cell_comm.around_target import melt_signal_target_data_frame
from cell2location.models import Cell2location, RegressionModel
from cell2location.models.simplified._cell2location_v3_no_factorisation_module import (
    LocationModelMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
)
from cell2location.models.simplified._cell2location_v3_no_mg_module import (
    LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelNoMGPyroModel,
)


def export_posterior(model, dataset):
    dataset = model.export_posterior(dataset, use_quantiles=True, add_to_obsm=["q50", "q001"])  # quantile 0.50
    dataset = model.export_posterior(
        dataset, use_quantiles=True, add_to_obsm=["q50"], sample_kwargs={"batch_size": 10}
    )  # quantile 0.50
    dataset = model.export_posterior(dataset, use_quantiles=True)  # default
    dataset = model.export_posterior(dataset, use_quantiles=True, sample_kwargs={"batch_size": 10})


def export_posterior_sc(model, dataset):
    dataset = model.export_posterior(dataset, use_quantiles=True, add_to_varm=["q50", "q001"])  # quantile 0.50
    dataset = model.export_posterior(
        dataset, use_quantiles=True, add_to_varm=["q50"], sample_kwargs={"batch_size": 10}
    )  # quantile 0.50
    dataset = model.export_posterior(dataset, use_quantiles=True)  # default
    dataset = model.export_posterior(dataset, use_quantiles=True, sample_kwargs={"batch_size": 10})


def test_cell2location():
    save_path = "./cell2location_model_test"
    if torch.cuda.is_available():
        use_gpu = int(torch.cuda.is_available())
        accelerator = "gpu"
    else:
        use_gpu = None
        accelerator = "cpu"
    dataset = synthetic_iid(n_labels=5)
    dataset.obsm["X_spatial"] = np.random.normal(0, 1, [dataset.n_obs, 2])
    RegressionModel.setup_anndata(dataset, labels_key="labels", batch_key="batch")

    # train regression model to get signatures of cell types
    sc_model = RegressionModel(dataset)
    # test full data training
    sc_model.train(max_epochs=1, accelerator=accelerator)
    # test minibatch training
    sc_model.train(max_epochs=1, batch_size=1000, accelerator=accelerator)
    # export the estimated cell abundance (summary of the posterior distribution)
    dataset = sc_model.export_posterior(dataset, sample_kwargs={"num_samples": 10})
    # test plot_QC
    sc_model.plot_QC()
    # test quantile export
    export_posterior_sc(sc_model, dataset)
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
    st_model.train(max_epochs=1, accelerator=accelerator)
    # export the estimated cell abundance (summary of the posterior distribution)
    # full data
    dataset = st_model.export_posterior(dataset, sample_kwargs={"num_samples": 10, "batch_size": st_model.adata.n_obs})
    # test quantile export
    export_posterior(st_model, dataset)
    st_model.plot_QC(summary_name="q05")
    ##  minibatches of locations  ##
    Cell2location.setup_anndata(dataset, batch_key="batch")
    st_model = Cell2location(dataset, cell_state_df=inf_aver, N_cells_per_location=30, detection_alpha=200)
    # test minibatch training
    st_model.train(max_epochs=1, batch_size=50, accelerator=accelerator)
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
    st_model.posterior_quantile(q=0.5, use_gpu=use_gpu)
    # test computing median
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

    ### test Messenger guide class ###
    Cell2location.setup_anndata(dataset, batch_key="batch")
    st_model = Cell2location(
        dataset,
        cell_state_df=inf_aver,
        N_cells_per_location=30,
        detection_alpha=200,
        create_autoguide_kwargs={"guide_class": AutoHierarchicalNormalMessenger},
    )
    # test minibatch training
    st_model.train(max_epochs=1, batch_size=50, accelerator=accelerator)
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
    # test quantile export
    dataset = st_model.export_posterior(
        dataset, use_quantiles=True, add_to_obsm=["q50"], sample_kwargs={"batch_size": 50}
    )  # only quantile 0.50 works with Messenger guide
    dataset = st_model.export_posterior(
        dataset,
        use_quantiles=True,
        add_to_obsm=["q50"],
    )  # only quantile 0.50 works with Messenger guide
    assert dataset.uns["mod"]["post_sample_q50"]["w_sf"].shape == (dataset.n_obs, dataset.obs["labels"].nunique())

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
    st_model.train(max_epochs=1, batch_size=20, accelerator=accelerator)
    st_model.train_aggressive(
        max_epochs=3,
        batch_size=20,
        plan_kwargs={"n_aggressive_epochs": 1, "n_aggressive_steps": 5},
        accelerator=accelerator,
        use_gpu=use_gpu,
    )
    # test hiding variables on the list
    var_list = ["locs.s_g_gene_add_alpha_e_inv"]
    for k, v in st_model.module.guide.named_parameters():
        k_in_vars = np.any([i in k for i in var_list])
        if k_in_vars:
            print(f"start {k} {v.requires_grad} {v.detach().cpu().numpy()}")
            v.requires_grad = False
            s_g_gene_add = v.detach().cpu().numpy()
    # test that normal training doesn't reactivate them
    st_model.train(max_epochs=1, batch_size=20, accelerator=accelerator)
    for k, v in st_model.module.guide.named_parameters():
        k_in_vars = np.any([i in k for i in var_list])
        if k_in_vars:
            print(f"train {k} {v.requires_grad} {v.detach().cpu().numpy()}")
            assert np.all(v.detach().cpu().numpy() == s_g_gene_add)
            v.requires_grad = False
    # test that aggressive training doesn't reactivate them
    st_model.train_aggressive(
        max_epochs=3,
        batch_size=20,
        plan_kwargs={"n_aggressive_epochs": 1, "n_aggressive_steps": 5},
        accelerator=accelerator,
        use_gpu=use_gpu,
    )
    for k, v in st_model.module.guide.named_parameters():
        k_in_vars = np.any([i in k for i in var_list])
        if k_in_vars:
            assert np.all(v.detach().cpu().numpy() == s_g_gene_add)
    # test computing median
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
    # test quantile export
    export_posterior(st_model, dataset)

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
    st_model.train(max_epochs=1, accelerator=accelerator)
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
    st_model.train(max_epochs=1, accelerator=accelerator)
    # export the estimated cell abundance (summary of the posterior distribution)
    # full data
    dataset = st_model.export_posterior(dataset, sample_kwargs={"num_samples": 10, "batch_size": st_model.adata.n_obs})

    # test compute_weighted_average_around_target
    normalisation_key = "detection_y_s"
    dataset.obsm[normalisation_key] = dataset.uns["mod"]["post_sample_q05"][normalisation_key]
    # average of other cell types
    compute_weighted_average_around_target(
        dataset,
        target_cell_type_quantile=0.995,
        source_cell_type_quantile=0.95,
        normalisation_quantile=0.999,
        sample_key="batch",
    )
    # average of genes
    compute_weighted_average_around_target(
        dataset,
        target_cell_type_quantile=0.995,
        source_cell_type_quantile=0.80,
        normalisation_quantile=0.999,
        normalisation_key=normalisation_key,
        genes_to_use_as_source=dataset.var_names,
        gene_symbols=None,
        sample_key="batch",
    )

    distance_bins = [
        [5, 50],
        [50, 100],
        [100, 150],
        [150, 200],
        [200, 250],
        [300, 350],
        [350, 400],
        [400, 450],
        [450, 500],
        [500, 550],
        [550, 600],
        [600, 650],
        [650, 700],
    ]
    weighted_avg_dict = dict()
    for distance_bin in distance_bins:
        # average of other cell types
        compute_weighted_average_around_target(
            dataset,
            target_cell_type_quantile=0.995,
            source_cell_type_quantile=0.95,
            normalisation_quantile=0.999,
            distance_bin=distance_bin,
            sample_key="batch",
        )
        # average of genes
        weighted_avg_dict[str(distance_bin)] = compute_weighted_average_around_target(
            dataset,
            target_cell_type_quantile=0.995,
            source_cell_type_quantile=0.80,
            normalisation_quantile=0.999,
            normalisation_key=normalisation_key,
            genes_to_use_as_source=dataset.var_names,
            gene_symbols=None,
            distance_bin=distance_bin,
            sample_key="batch",
        )
    melt_signal_target_data_frame(weighted_avg_dict, distance_bins)
