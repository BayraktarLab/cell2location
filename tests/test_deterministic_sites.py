import anndata as ad
import numpy as np
import pandas as pd
import pyro
import pytest
from scipy import sparse
from scvi.dataloaders import AnnDataLoader

from cell2location.models import Cell2location, Cell2location_WTA
from cell2location.models._cell2location_module import (
    LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
)
from cell2location.models.simplified._cell2location_v3_no_factorisation_module import (
    LocationModelMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
)
from cell2location.models.simplified._cell2location_v3_no_mg_module import (
    LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelNoMGPyroModel,
)


SITE_NAME = "u_sf_mRNA_factors"


def make_random_problem(n_obs=8, n_genes=16, n_factors=3, seed=0):
    rng = np.random.default_rng(seed)
    genes = pd.Index([f"gene_{i}" for i in range(n_genes)])
    factors = [f"cell_type_{i}" for i in range(n_factors)]
    signatures = rng.gamma(2.0, 1.0, size=(n_genes, n_factors)).astype(np.float32)
    counts = rng.poisson(2.0, size=(n_obs, n_genes)).astype(np.float32)
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame(index=[f"location_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=genes),
    )
    return adata, pd.DataFrame(signatures, index=genes, columns=factors)


def unwrap_base_distribution(distribution):
    while hasattr(distribution, "base_dist"):
        distribution = distribution.base_dist
    return distribution


@pytest.mark.parametrize(
    "model_class",
    [
        LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
        LocationModelMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
        LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelNoMGPyroModel,
    ],
)
def test_mrna_factors_delta_has_linear_shape(model_class):
    n_obs, n_factors = 8, 3
    adata, cell_state_df = make_random_problem(n_obs=n_obs, n_factors=n_factors)
    Cell2location.setup_anndata(adata)
    model = Cell2location(
        adata,
        cell_state_df=cell_state_df,
        model_class=model_class,
        N_cells_per_location=3,
        detection_alpha=20,
    )
    loader = AnnDataLoader(model.adata_manager, shuffle=False, batch_size=n_obs)
    model_args, model_kwargs = model.module._get_fn_args_from_batch(next(iter(loader)))

    trace = pyro.poutine.trace(model.module.model).get_trace(*model_args, **model_kwargs)
    node = trace.nodes[SITE_NAME]
    delta = unwrap_base_distribution(node["fn"])

    assert tuple(node["value"].shape) == (n_obs, n_factors)
    assert tuple(delta.v.shape) == (n_obs, n_factors)
    assert delta.v.numel() == n_obs * n_factors


def test_wta_mrna_factors_delta_has_linear_shape():
    n_obs, n_factors = 8, 3
    adata, cell_state_df = make_random_problem(n_obs=n_obs, n_factors=n_factors)
    adata.obs["nuclei"] = 3.0
    adata.obsm["negative_probes"] = np.ones((n_obs, 2), dtype=np.float32)
    Cell2location_WTA.setup_anndata(adata, nuclei_key="nuclei", neg_probes_key="negative_probes")
    model = Cell2location_WTA(
        adata,
        cell_state_df=cell_state_df,
        detection_alpha=20,
    )
    loader = AnnDataLoader(model.adata_manager, shuffle=False, batch_size=n_obs)
    model_args, model_kwargs = model.module._get_fn_args_from_batch(next(iter(loader)))

    trace = pyro.poutine.trace(model.module.model).get_trace(*model_args, **model_kwargs)
    node = trace.nodes[SITE_NAME]
    delta = unwrap_base_distribution(node["fn"])

    assert tuple(node["value"].shape) == (n_obs, n_factors)
    assert tuple(delta.v.shape) == (n_obs, n_factors)
    assert delta.v.numel() == n_obs * n_factors
