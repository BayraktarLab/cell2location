import numpy as np
from scvi.data import synthetic_iid

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
    dataset = synthetic_iid(n_labels=5, run_setup_anndata=False)
    RegressionModel.setup_anndata(dataset, labels_key="labels", batch_key="batch")

    # train regression model to get signatures of cell types
    sc_model = RegressionModel(dataset)
    # test full data training
    sc_model.train(max_epochs=1)
    # test minibatch training
    sc_model.train(max_epochs=1, batch_size=1000)
    # export the estimated cell abundance (summary of the posterior distribution)
    dataset = sc_model.export_posterior(dataset, sample_kwargs={"num_samples": 10})
    # test plot_QC
    sc_model.plot_QC()
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
    ##  full data  ##
    st_model = Cell2location(dataset, cell_state_df=inf_aver, N_cells_per_location=30, detection_alpha=200)
    # test full data training
    st_model.train(max_epochs=1)
    # export the estimated cell abundance (summary of the posterior distribution)
    # full data
    dataset = st_model.export_posterior(dataset, sample_kwargs={"num_samples": 10, "batch_size": st_model.adata.n_obs})
    ##  minibatches of locations  ##
    st_model = Cell2location(dataset, cell_state_df=inf_aver, N_cells_per_location=30, detection_alpha=200)
    # test minibatch training
    st_model.train(max_epochs=1, batch_size=50)
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
    st_model.posterior_quantile(q=0.5)
    # test computing expected expression per cell type
    st_model.module.model.compute_expected_per_cell_type(st_model.samples["post_sample_q05"], st_model.adata)
    ### test amortised inference with default cell2location model ###
    ##  full data  ##
    st_model = Cell2location(
        dataset,
        cell_state_df=inf_aver,
        N_cells_per_location=30,
        detection_alpha=200,
        amortised=True,
        encoder_mode="multiple",
    )
    # test minibatch training
    st_model.train(max_epochs=1, batch_size=50)
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
    st_model = Cell2location(
        dataset,
        cell_state_df=inf_aver,
        N_cells_per_location=30,
        detection_alpha=200,
        model_class=LocationModelMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
    )
    # test full data training
    st_model.train(max_epochs=1)
    # export the estimated cell abundance (summary of the posterior distribution)
    # full data
    dataset = st_model.export_posterior(dataset, sample_kwargs={"num_samples": 10, "batch_size": st_model.adata.n_obs})
    ##  no w_sf factorisation  ##
    st_model = Cell2location(
        dataset,
        cell_state_df=inf_aver,
        N_cells_per_location=30,
        detection_alpha=200,
        model_class=LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelNoMGPyroModel,
    )
    # test full data training
    st_model.train(max_epochs=1)
    # export the estimated cell abundance (summary of the posterior distribution)
    # full data
    st_model.export_posterior(dataset, sample_kwargs={"num_samples": 10, "batch_size": st_model.adata.n_obs})
