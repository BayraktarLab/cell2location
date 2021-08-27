# -*- coding: utf-8 -*-
r"""Run full pipeline for locating cells with cell2location model."""

import gc
import os
import pickle
import time
from os import mkdir

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import theano

import cell2location.models as models
import cell2location.plt as c2lpl
from cell2location.cluster_averages import compute_cluster_averages, select_features
from cell2location.models.base.pymc3_loc_model import Pymc3LocModel


def save_plot(path, filename, extension="png"):
    r"""Save current plot to `path` as `filename` with `extension`"""

    plt.savefig(path + filename + "." + extension)
    # fig.clear()
    # plt.close()


def run_cell2location(
    sc_data,
    sp_data,
    model_name=None,
    verbose=True,
    show_locations=False,
    return_all=True,
    summ_sc_data_args={
        "cluster_col": "annotation_1",
        "selection": "cluster_specificity",
        "selection_specificity": 0.07,
    },
    train_args={
        "n_iter": 20000,
        "learning_rate": 0.005,
        "sample_prior": False,
        "readable_var_name_col": None,
        "sample_name_col": None,
    },
    model_kwargs={},
    posterior_args={"n_samples": 1000},
    export_args={
        "path": "./results",
        "save_model": False,
        "save_spatial_plots": True,
        "run_name_suffix": "",
        "scanpy_coords_name": "coords",
    },
):
    r"""Run cell2location model pipeline: train the model, sample prior and posterior,
    export results and save diagnostic plots

    Briefly, cell2location is a Bayesian model, which estimates absolute cell density of cell types by
    decomposing mRNA counts :math:`d_{s,g}` of each gene :math:`g = {1, .., G}` at locations :math:`s = {1, .., S}`
    into a set of predefined reference signatures of cell types :math:`g_{f,g}`.

    The cell2location software comes with two implementations for this estimation step:
    1) a statistical method based on Negative Binomial regression (see `cell2location.run_regression`);
    2) hard-coded computation of per-cluster average mRNA counts for individual genes
      (provided anndata object to this function, see below).

    Approximate Variational Inference is used to estimate all model parameters,
    implemented in the pymc3 framework, which supports GPU acceleration.

    Anndata object with exported results, W weights representing cell abundance for each cell type,
    and the trained model object are saved to `export_args['path']`

    .. note:: This pipeline is not specific to a particular model so please look up the relevant parameters
        in model documentation (the default is LocationModelLinearDependentWMultiExperiment).

    Parameters
    ----------
    sc_data :
        anndata object with single cell / nucleus data,
        or pd.DataFrame with genes in rows and cell type signatures (factors) in columns.
    sp_data :
        anndata object with spatial data, variable names should match sc_data
    model_name :
        model class object or None. If none, the default single-sample or multi-sample model is selected
        depending on the number of samples detected in in the sp_data.obs columns
        specified by `train_args['sample_name_col']`.
    verbose :
        boolean, print diagnostic messages? (Default value = True)
    show_locations :
        boolean, print the plot of factor locations? If False the plots are saved but not shown.
        This reduces notebook size. (Default value = False)
    return_all :
        boolean, return the model and annotated `sp_data`. If True, both are saved but not returned? (Default value = True)
    summ_sc_data_args :
        arguments for summarising single cell data

        * **cluster_col** - name of sc_data.obs column containing cluster annotations
        * **which_genes** - select intersect or union genes between single cell and spatial? (Default: intersect)
        * **selection** - select highly variable genes in sc_data? (Default: None - do not select), available options:

            * None
            * **cluster_specificity** use marker specificity to clusters to select most informative genes
              (see **selection_specificity** below)
            * **high_cv** use rank by coefficient of variation (variance / mean)
            * **cluster_markers** use cluster markers (derived using sc.tl.rank_genes_groups)
            * **AutoGeneS** use https://github.com/theislab/AutoGeneS
            * **select_n** - how many variable genes to select? Used only when selection is not None (Default: 5000).
            * **select_n_AutoGeneS** - how many variable genes tor select with AutoGeneS (lower than select_n)? (Default: 1000).

        * **selection_specificity** expression specificity cut-off (default=0.1), expression signatures of cell types
          are normalised to sum to 1 gene-wise, then a cutoff is applied. Decrease cutoff to include more genes.

    train_args :
        arguments for training methods. See help(c2l.LocationModelLinearDependentWMultiExperiment) for more details

        * **mode** - "normal" or "tracking" parameters? (Default: normal)
        * **use_raw** - extract data from RAW slot of anndata? Applies only to spatial data, not single cell reference.
          For reference sc_data.raw is always used. (Default: True)
        * **data_type** - use arrays with which precision? (Default: 'float32').
        * **n_iter** - number of training iterations (Default: 20000).
        * **learning_rate** - ADAM optimiser learning rate (Default: 0.005).
          If the training is very unstable try reducing the learning rate.
        * **total_grad_norm_constraint** - ADAM optimiser total_grad_norm_constraint (Default: 200).
          Prevents exploding gradient problem, of the training is very unstable (jumps to Inf/NaN loss) try reducing this parameter.
        * **method** - which method to use to find posterior. Use default 'advi' unless you know what you are doing.
        * **sample_prior** - use sampling from the prior to evaluate how reasonable priors are for your data. (Default: False)
          This is not essential for 10X Visium data but can be very useful for troubleshooting when
          the model fails training with NaN or Inf likelihood / ELBO loss.
          It is essential to do this first to choose priors appropriate for any other technology.
          Sample from the prior should be in the same range as data, span similar number of orders of magnitude,
          and ideally be weakly informative: results/plots/evaluating_prior.png plot should looks like a very wide diagonal.

        Caution
        -------
            Sampling the prior takes a lot of RAM (10s-100s of GB) depending on data size so if needed for troubleshooting - reduce the number of locations in needed.

        * **n_prior_samples** - number of prior samples. The more the better but also takes a lot of RAM. (Default: 10)
        * **n_restarts** - number of training restarts to evaluate stability (Default: 2)
        * **n_type** - type of restart training: 'restart' 'cv' 'bootstrap' (Default: restart)
          (see help(cell2location.LocationModel.fit_advi_iterative) for details)
        * **tracking_every**, **tracking_n_samples** - parameters for "tracking" mode:
          Posterior samples are saved after 'tracking_every' iterations (Default: 1000),
          the process is repeated 'tracking_n_samples' times (Default: 50)
        * **readable_var_name_col** - column in sp_data.var that contains readable gene ID (e.g. HGNC symbol) (Default: None)
        * **sample_name_col** - column in sp_data.obs that contains sample / slide ID (e.g. Visium section),
          for plotting W cell locations from a multi-sample object correctly (Default: None)
        * **fact_names** - optional list of factor names, by default taken from cluster names in sc_data.

    posterior_args :
        arguments for sampling posterior
        The model generates samples from the posterior distribution of each parameter
        to compute mean, SD, 5% and 95% quantiles - stored in `mod.samples` and exported in `adata.uns['mod']`.

        * **n_samples** - number of samples to generate (the more the better but you run out
          of the GPU memory if too many). (Default: 1000)
        * **evaluate_stability_align** - when generating the model stability plots, align different restarts? (Default: False)
        * **mean_field_slot** - Posterior of all parameters is sampled only from one training restart due to stability of the model. -
          which training restart to use? (Default: 'init_1')
    export_args :
        arguments for exporting results

        * **path** - file path where to save results (Default: "./results")
        * **plot_extension** - file extension of saved plots (Default: "png")
        * **scanpy_plot_vmax** - 'p99.2', 'scanpy_plot_size': 1.3 - scanpy.pl.spatial plottin settings
        * **save_model** - boolean, save trained model? Could be useful but also takes up 10s of GB of disk space. (Default: False)
        * **run_name_suffix** - optinal suffix to modify the name the run. (Default: '')
        * **export_q05** - boolean, save plots of 5% quantile of parameters. (Default: True)
        * **scanpy_coords_name** - sp_data.obsm entry that stores X and Y coordinates of each location.
          If None - no spatial plot is produced.
        * **img_key** - which image to use for scanpy plotting ('hires', 'lowres', None)
    model_kwargs :
        Keyword arguments for the model class. See the list of relevant arguments for CoLocationModelNB4V2. (Default value = {})

    Returns
    -------
    dict
        Results as a dictionary, use dict.keys() to find the elements. Results are saved to `export_args['path']`.

    """

    # set default parameters
    d_summ_sc_data_args = {
        "cluster_col": "annotation_1",
        "which_genes": "intersect",
        "selection": None,
        "select_n": 5000,
        "select_n_AutoGeneS": 1000,
        "selection_specificity": 0.1,
        "cluster_markers_kwargs": {},
    }

    d_train_args = {
        "mode": "normal",
        "use_raw": True,
        "data_type": "float32",
        "n_iter": 20000,
        "learning_rate": 0.005,
        "total_grad_norm_constraint": 200,
        "method": "advi",
        "sample_prior": False,
        "n_prior_samples": 10,
        "n_restarts": 2,
        "n_type": "restart",
        "tracking_every": 1000,
        "tracking_n_samples": 50,
        "readable_var_name_col": None,
        "sample_name_col": None,
        "fact_names": None,
        "minibatch_size": None,
    }

    d_posterior_args = {"n_samples": 1000, "evaluate_stability_align": False, "mean_field_slot": "init_1"}

    d_export_args = {
        "path": "./results",
        "plot_extension": "png",
        "scanpy_plot_vmax": "p99.2",
        "scanpy_plot_size": 1.3,
        "save_model": False,
        "save_spatial_plots": True,
        "run_name_suffix": "",
        "export_q05": True,
        "export_mean": False,
        "scanpy_coords_name": "spatial",
        "img_key": "hires",
    }

    # replace defaults with parameters supplied
    for k in summ_sc_data_args.keys():
        d_summ_sc_data_args[k] = summ_sc_data_args[k]
    summ_sc_data_args = d_summ_sc_data_args
    for k in train_args.keys():
        d_train_args[k] = train_args[k]
    train_args = d_train_args
    for k in posterior_args.keys():
        d_posterior_args[k] = posterior_args[k]
    posterior_args = d_posterior_args
    for k in export_args.keys():
        d_export_args[k] = export_args[k]
    export_args = d_export_args

    theano.config.allow_gc = True

    # start timing
    start = time.time()

    if np.sum(sp_data.obs_names.duplicated()) > 0:
        raise ValueError("Make sure location names are unique (`sp_data.obs_names`)")

    sp_data = sp_data.copy()

    ####### Summarising single cell clusters #######
    if verbose:
        print("### Summarising single cell clusters ###")

    if not isinstance(sc_data, pd.DataFrame):
        # if scanpy compute cluster averages
        cell_state_df = compute_cluster_averages(sc_data, cluster_col=summ_sc_data_args["cluster_col"])
        obs = sc_data.obs
    else:
        # if dataframe assume signature with matching .index to sp_data.var_names
        cell_state_df = sc_data.copy()
        obs = cell_state_df

    # select features if requested
    if summ_sc_data_args["selection"] == "high_cv":

        cv = cell_state_df.var(1) / cell_state_df.mean(1)
        cv = cv.sort_values(ascending=False)
        cell_state_df = cell_state_df.loc[cv[np.arange(summ_sc_data_args["select_n"])].index, :]

    elif summ_sc_data_args["selection"] == "cluster_markers":
        if isinstance(sc_data, pd.DataFrame):
            raise ValueError(
                'summ_sc_data_args["selection"] = "cluster_markers" can only be used with type(sc_data) Anndata'
            )

        sel_feat = select_features(
            sc_data,
            summ_sc_data_args["cluster_col"],
            n_features=summ_sc_data_args["select_n"],
            use_raw=train_args["use_raw"],
            sc_kwargs=summ_sc_data_args["cluster_markers_kwargs"],
        )
        cell_state_df = cell_state_df.loc[cell_state_df.index.isin(sel_feat), :]

    elif summ_sc_data_args["selection"] == "cluster_specificity":

        # selecting most informative genes based on specificity
        cell_state_df_norm = (cell_state_df.T / cell_state_df.sum(1)).T
        cell_state_df_norm = cell_state_df_norm > summ_sc_data_args["selection_specificity"]
        sel_feat = cell_state_df_norm.index[cell_state_df_norm.sum(1) > 0]
        cell_state_df = cell_state_df.loc[cell_state_df.index.isin(sel_feat), :]

    elif summ_sc_data_args["selection"] == "AutoGeneS":

        cv = cell_state_df.var(1) / cell_state_df.mean(1)
        cv = cv.sort_values(ascending=False)
        cell_state_df = cell_state_df.loc[cv[np.arange(summ_sc_data_args["select_n"])].index, :]

        from autogenes import AutoGenes

        ag = AutoGenes(cell_state_df.T)
        ag.run(ngen=5000, seed=0, nfeatures=summ_sc_data_args["select_n_AutoGeneS"], mode="fixed")
        pareto = ag.pareto
        print(ag.plot(size="large", weights=(1, -1)))
        # We then pick one solution and filter its corresponding marker genes:
        cell_state_df = cell_state_df[pareto[len(pareto) - 1]]  # select the solution with min correlation

    elif summ_sc_data_args["selection"] is not None:
        raise ValueError("summ_sc_data_args['selection'] can be only None, high_cv, cluster_markers or AutoGeneS")

    # extract data as a dense matrix
    if train_args["use_raw"]:
        if scipy.sparse.issparse(sp_data.raw.X):
            X_data = np.array(sp_data.raw.X.toarray())
        else:
            X_data = np.array(sp_data.raw.X)
    else:
        if scipy.sparse.issparse(sp_data.raw.X):
            X_data = np.array(sp_data.X.toarray())
        else:
            X_data = np.array(sp_data.X)

    # Filter cell states and X_data to common genes
    sp_ind = sp_data.var_names.isin(cell_state_df.index)
    if np.sum(sp_ind) == 0:
        raise ValueError("No overlapping genes found, check that `sc_data` and `sp_data` use the same variable names")
    X_data = X_data[:, sp_ind]
    cell_state_df = cell_state_df.loc[sp_data.var_names[sp_ind], :]

    # prepare cell state matrix
    cell_state_mat = cell_state_df.values

    # if factor names not provided use cluster names
    if train_args["fact_names"] is None:
        fact_names = cell_state_df.columns
    else:  # useful for unobserved factor models
        fact_names = train_args["fact_names"]

    if train_args["sample_name_col"] is None:
        sp_data.obs["sample"] = "sample"
        train_args["sample_name_col"] = "sample"

    if train_args["readable_var_name_col"] is not None:
        readable_var_name_col = sp_data.var[train_args["readable_var_name_col"]][sp_ind]
    else:
        readable_var_name_col = None

    if train_args["minibatch_size"] is not None:
        model_kwargs["minibatch_size"] = train_args["minibatch_size"]

    ####### Creating model #######
    # choose model when not specified
    if model_name is None:
        sample_name = sp_data.obs[train_args["sample_name_col"]]
        sample_n = len(np.unique(sample_name))
        if sample_n < sp_data.shape[0]:
            Model = models.pymc3.LocationModelLinearDependentWMultiExperimentLocationBackgroundNormGeneAlpha
            model_name = "LocationModelLinearDependentWMultiExperimentLocationBackgroundNormGeneAlpha"
        else:
            ValueError(
                "train_args['sample_name_col'] points to non-existing column or the number of samples(batches) is equal to the number of observations `adata.n_obs`"
            )
    else:  # use supplied class
        Model = model_name

    n_exper = len(sp_data.obs[train_args["sample_name_col"]].unique())
    if n_exper == X_data.shape[0]:
        raise ValueError(
            "The number of samples is equal to the number of locations, aborting... (check 'sample_name_col')"
        )

    if verbose:
        print("### Creating model ### - time " + str(np.around((time.time() - start) / 60, 2)) + " min")
    mod = Model(
        cell_state_mat,
        X_data,
        data_type=train_args["data_type"],
        n_iter=train_args["n_iter"],
        learning_rate=train_args["learning_rate"],
        total_grad_norm_constraint=train_args["total_grad_norm_constraint"],
        verbose=verbose,
        var_names=sp_data.var_names[sp_ind],
        var_names_read=readable_var_name_col,
        obs_names=sp_data.obs_names,
        fact_names=fact_names,
        sample_id=sp_data.obs[train_args["sample_name_col"]],
        **model_kwargs,
    )

    ####### Print run name #######
    run_name = f"{mod.__class__.__name__}_{n_exper}experiments_{mod.n_fact}clusters_{mod.n_obs}locations_{mod.n_var}genes{export_args['run_name_suffix']}"

    print("### Analysis name: " + run_name)  # analysis name is always printed

    # create the export directory
    path = export_args["path"] + run_name + "/"
    if not os.path.exists(path):
        mkdir(path)

    fig_path = path + "plots/"
    if not os.path.exists(fig_path):
        mkdir(fig_path)

    ####### Sampling prior #######
    if train_args["sample_prior"]:
        if verbose:
            print("### Sampling prior ###")

        mod.sample_prior(samples=train_args["n_prior_samples"])

        # plot & save plot
        mod.plot_prior_vs_data()

        plt.tight_layout()
        save_plot(fig_path, filename="evaluating_prior", extension=export_args["plot_extension"])
        plt.close()

    ####### Training model #######
    if verbose:
        print("### Training model ###")
    if train_args["mode"] == "normal":
        mod.train(
            n=train_args["n_restarts"], method=train_args["method"], n_type=train_args["n_type"], progressbar=verbose
        )

    elif train_args["mode"] == "tracking":
        mod.verbose = False
        if isinstance(mod, Pymc3LocModel):
            mod.track_parameters(
                n=train_args["n_restarts"],
                every=train_args["tracking_every"],
                n_samples=train_args["tracking_n_samples"],
                n_type=train_args["n_type"],
                df_node_name1="nUMI_factors",
                df_node_df_name1="spot_factors_df",
                df_prior_node_name1="spot_fact_mu_hyp",
                df_prior_node_name2="spot_fact_sd_hyp",
                mu_node_name="mu",
                data_node="X_data",
                extra_df_parameters=["spot_add"],
                sample_type="post_sample_means",
            )

        else:  # TODO add isinstance(mod, PyroModel)
            mod.train(
                n=train_args["n_restarts"], method=train_args["method"], n_type=train_args["n_type"], tracking=True
            )

    else:
        raise ValueError("train_args['mode'] can be only 'normal' or 'tracking'")

    theano.config.compute_test_value = "ignore"
    ####### Evaluate stability of training #######
    if train_args["n_restarts"] > 1:
        n_plots = train_args["n_restarts"] - 1
        ncol = int(np.min((n_plots, 3)))
        nrow = np.ceil(n_plots / ncol)
        plt.figure(figsize=(5 * nrow, 5 * ncol))
        mod.evaluate_stability(n_samples=posterior_args["n_samples"], align=posterior_args["evaluate_stability_align"])

        plt.tight_layout()
        save_plot(fig_path, filename="evaluate_stability", extension=export_args["plot_extension"])
        plt.close()

    ####### Sampling posterior #######
    if verbose:
        print("### Sampling posterior ### - time " + str(np.around((time.time() - start) / 60, 2)) + " min")
    mod.sample_posterior(
        node="all",
        n_samples=posterior_args["n_samples"],
        save_samples=False,
        mean_field_slot=posterior_args["mean_field_slot"],
    )

    # evaluate predictive accuracy
    mod.compute_expected()

    ####### Export summarised posterior & Saving results #######
    if verbose:
        print("### Saving results ###")

    # save W cell locations (cell density)
    # convert cell location parameters (cell density) to a dataframe
    mod.sample2df(node_name="spot_factors")
    # add cell location parameters to `sp_data.obs`
    sp_data = mod.annotate_spot_adata(sp_data)
    # save
    mod.spot_factors_df.to_csv(path + "W_cell_density.csv")
    if export_args["export_q05"]:
        mod.spot_factors_q05.to_csv(path + "W_cell_density_q05.csv")

    # convert cell location parameters (mRNA count) to a dataframe, see help(mod.sample2df) for details
    mod.sample2df(node_name="nUMI_factors")

    # add cell location parameters to `sp_data.obs`
    sp_data = mod.annotate_spot_adata(sp_data)

    # save W cell locations (mRNA count)
    mod.spot_factors_df.to_csv(path + "W_mRNA_count.csv")
    if export_args["export_q05"]:
        mod.spot_factors_q05.to_csv(path + "W_mRNA_count_q05.csv")

    # add posterior of all parameters to `sp_data.uns['mod']`
    mod.fact_filt = None
    sp_data = mod.export2adata(adata=sp_data, slot_name="mod")
    sp_data.uns["mod"]["fact_names"] = list(sp_data.uns["mod"]["fact_names"])
    sp_data.uns["mod"]["var_names"] = list(sp_data.uns["mod"]["var_names"])
    sp_data.uns["mod"]["obs_names"] = list(sp_data.uns["mod"]["obs_names"])

    # save spatial anndata with exported posterior
    sp_data.write(filename=path + "sp.h5ad", compression="gzip")

    # save model object and related annotations
    if export_args["save_model"]:
        # save the model and other objects
        res_dict = {
            "mod": mod,
            "sp_data": sp_data,
            "sc_obs": obs,
            "model_name": model_name,
            "summ_sc_data_args": summ_sc_data_args,
            "train_args": train_args,
            "posterior_args": posterior_args,
            "export_args": export_args,
            "run_name": run_name,
            "run_time": str(np.around((time.time() - start) / 60, 2)) + " min",
        }
        pickle.dump(res_dict, file=open(path + "model_.p", "wb"))

    else:
        # just save the settings
        res_dict = {
            "sc_obs": obs,
            "model_name": model_name,
            "summ_sc_data_args": summ_sc_data_args,
            "train_args": train_args,
            "posterior_args": posterior_args,
            "export_args": export_args,
            "run_name": run_name,
            "run_time": str(np.around((time.time() - start) / 60, 2)) + " min",
        }
        pickle.dump(res_dict, file=open(path + "model_.p", "wb"))
        # add model for returning
        res_dict["mod"] = mod

    ####### Plotting #######
    if verbose:
        print("### Ploting results ###")

    # Show training history #
    if verbose:
        mod.plot_history(0)
    else:
        mod.plot_history(0)

    plt.tight_layout()
    save_plot(fig_path, filename="training_history_all", extension=export_args["plot_extension"])
    plt.close()

    if verbose:
        mod.plot_history(int(np.ceil(train_args["n_iter"] * 0.2)))
    else:
        mod.plot_history(int(np.ceil(train_args["n_iter"] * 0.2)))

    plt.tight_layout()
    save_plot(fig_path, filename="training_history_without_first_20perc", extension=export_args["plot_extension"])
    plt.close()

    # Predictive accuracy
    try:
        mod.plot_posterior_mu_vs_data()
        plt.tight_layout()
        save_plot(fig_path, filename="data_vs_posterior_mean", extension=export_args["plot_extension"])
        plt.close()
    except Exception as e:
        print("Some error in plotting `mod.plot_posterior_mu_vs_data()`\n " + str(e))

    ####### Plotting posterior of W / cell locations #######
    if verbose:
        print("### Plotting posterior of W / cell locations ###")

    data_samples = sp_data.obs[train_args["sample_name_col"]].unique()
    cluster_plot_names = pd.Series([i[17:] for i in mod.spot_factors_df.columns])

    if export_args["save_spatial_plots"]:
        try:
            for s in data_samples:
                # if slots needed to generate scanpy plots are present, scanpy:
                sc_spatial_present = np.any(np.isin(list(sp_data.uns.keys()), ["spatial"]))

                if sc_spatial_present:

                    sc.settings.figdir = fig_path + "spatial/"
                    os.makedirs(fig_path + "spatial/", exist_ok=True)

                    with matplotlib.rc_context({"axes.facecolor": "black"}):

                        s_ind = sp_data.obs[train_args["sample_name_col"]] == s
                        s_keys = list(sp_data.uns["spatial"].keys())
                        s_spatial = np.array(s_keys)[[s in i for i in s_keys]][0]

                        if export_args["export_mean"]:
                            # Visualize cell type locations - mRNA_count = nUMI #####
                            # making copy to transform to log & assign nice names
                            clust_names_orig = ["mean_nUMI_factors" + i for i in sp_data.uns["mod"]["fact_names"]]
                            clust_names = sp_data.uns["mod"]["fact_names"]

                            sp_data.obs[clust_names] = sp_data.obs[clust_names_orig]
                            fig = sc.pl.spatial(
                                sp_data[s_ind, :],
                                cmap="magma",
                                color=clust_names,
                                ncols=5,
                                library_id=s_spatial,
                                size=export_args["scanpy_plot_size"],
                                img_key=export_args["img_key"],
                                alpha_img=0,
                                vmin=0,
                                vmax=export_args["scanpy_plot_vmax"],
                                return_fig=True,
                                show=False,
                            )

                            fig.savefig(
                                f"{fig_path}/spatial/W_mRNA_count_mean_{s}_{export_args['scanpy_plot_vmax']}"
                                f".{export_args['plot_extension']}",
                                bbox_inches="tight",
                            )
                            # fig.clear()
                            plt.close()
                            if show_locations:
                                plt.show()

                            fig = sc.pl.spatial(
                                sp_data[s_ind, :],
                                cmap="magma",
                                color=clust_names,
                                ncols=5,
                                library_id=s_spatial,
                                size=export_args["scanpy_plot_size"],
                                img_key=export_args["img_key"],
                                alpha_img=1,
                                vmin=0,
                                vmax=export_args["scanpy_plot_vmax"],
                                show=False,
                                return_fig=True,
                            )
                            fig.savefig(
                                f"{fig_path}/spatial/histo_W_mRNA_count_mean_{s}_{export_args['scanpy_plot_vmax']}"
                                f".{export_args['plot_extension']}",
                                bbox_inches="tight",
                            )
                            # fig.clear()
                            plt.close()

                            # Visualize cell type locations #####
                            # making copy to transform to log & assign nice names
                            clust_names_orig = ["mean_spot_factors" + i for i in sp_data.uns["mod"]["fact_names"]]
                            clust_names = sp_data.uns["mod"]["fact_names"]
                            sp_data.obs[clust_names] = sp_data.obs[clust_names_orig]

                            fig = sc.pl.spatial(
                                sp_data[s_ind, :],
                                cmap="magma",
                                color=clust_names,
                                ncols=5,
                                library_id=s_spatial,
                                size=export_args["scanpy_plot_size"],
                                img_key=export_args["img_key"],
                                alpha_img=0,
                                vmin=0,
                                vmax=export_args["scanpy_plot_vmax"],
                                show=False,
                                return_fig=True,
                            )
                            fig.savefig(
                                f"{fig_path}/spatial/W_cell_density_mean_{s}_{export_args['scanpy_plot_vmax']}."
                                f"{export_args['plot_extension']}",
                                bbox_inches="tight",
                            )
                            # fig.clear()
                            plt.close()

                            fig = sc.pl.spatial(
                                sp_data[s_ind, :],
                                cmap="magma",
                                color=clust_names,
                                ncols=5,
                                library_id=s_spatial,
                                size=export_args["scanpy_plot_size"],
                                img_key=export_args["img_key"],
                                alpha_img=1,
                                vmin=0,
                                vmax=export_args["scanpy_plot_vmax"],
                                show=False,
                                return_fig=True,
                            )
                            plt.savefig(
                                f"{fig_path}/spatial/histo_W_cell_density_mean_{s}_{export_args['scanpy_plot_vmax']}"
                                f".{export_args['plot_extension']}",
                                bbox_inches="tight",
                            )
                            # fig.clear()
                            plt.close()

                        if export_args["export_q05"]:
                            # Visualize cell type locations - mRNA_count = nUMI #####
                            # making copy to transform to log & assign nice names
                            clust_names_orig = ["q05_nUMI_factors" + i for i in sp_data.uns["mod"]["fact_names"]]
                            clust_names = sp_data.uns["mod"]["fact_names"]

                            sp_data.obs[clust_names] = sp_data.obs[clust_names_orig]
                            fig = sc.pl.spatial(
                                sp_data[s_ind, :],
                                cmap="magma",
                                color=clust_names,
                                ncols=5,
                                library_id=s_spatial,
                                size=export_args["scanpy_plot_size"],
                                img_key=export_args["img_key"],
                                alpha_img=0,
                                vmin=0,
                                vmax=export_args["scanpy_plot_vmax"],
                                show=False,
                                return_fig=True,
                            )
                            plt.savefig(
                                f"{fig_path}/spatial/W_mRNA_count_q05_{s}_{export_args['scanpy_plot_vmax']}"
                                f".{export_args['plot_extension']}",
                                bbox_inches="tight",
                            )
                            # fig.clear()
                            plt.close()

                            fig = sc.pl.spatial(
                                sp_data[s_ind, :],
                                cmap="magma",
                                color=clust_names,
                                ncols=5,
                                library_id=s_spatial,
                                size=export_args["scanpy_plot_size"],
                                img_key=export_args["img_key"],
                                alpha_img=1,
                                vmin=0,
                                vmax=export_args["scanpy_plot_vmax"],
                                show=False,
                                return_fig=True,
                            )
                            plt.savefig(
                                f"{fig_path}/spatial/histo_W_mRNA_count_q05_{s}_{export_args['scanpy_plot_vmax']}"
                                f".{export_args['plot_extension']}",
                                bbox_inches="tight",
                            )
                            # fig.clear()
                            plt.close()

                            # Visualize cell type locations #####
                            # making copy to transform to log & assign nice names
                            clust_names_orig = ["q05_spot_factors" + i for i in sp_data.uns["mod"]["fact_names"]]
                            clust_names = sp_data.uns["mod"]["fact_names"]
                            sp_data.obs[clust_names] = sp_data.obs[clust_names_orig]

                            fig = sc.pl.spatial(
                                sp_data[s_ind, :],
                                cmap="magma",
                                color=clust_names,
                                ncols=5,
                                library_id=s_spatial,
                                size=export_args["scanpy_plot_size"],
                                img_key=export_args["img_key"],
                                alpha_img=0,
                                vmin=0,
                                vmax=export_args["scanpy_plot_vmax"],
                                show=False,
                                return_fig=True,
                            )
                            plt.savefig(
                                f"{fig_path}/spatial/W_cell_density_q05_{s}_{export_args['scanpy_plot_vmax']}"
                                f".{export_args['plot_extension']}",
                                bbox_inches="tight",
                            )
                            # fig.clear()
                            plt.close()

                            fig = sc.pl.spatial(
                                sp_data[s_ind, :],
                                cmap="magma",
                                color=clust_names,
                                ncols=5,
                                library_id=s_spatial,
                                size=export_args["scanpy_plot_size"],
                                img_key=export_args["img_key"],
                                alpha_img=1,
                                vmin=0,
                                vmax=export_args["scanpy_plot_vmax"],
                                show=False,
                                return_fig=True,
                            )
                            plt.savefig(
                                f"{fig_path}/spatial/histo_W_cell_density_q05_{s}_{export_args['scanpy_plot_vmax']}"
                                f".{export_args['plot_extension']}",
                                bbox_inches="tight",
                            )
                            # fig.clear()
                            plt.close()

                else:

                    # if coordinates exist plot
                    if export_args["scanpy_coords_name"] is not None:
                        # move spatial coordinates to obs for compatibility with our plotter
                        sp_data.obs["imagecol"] = sp_data.obsm[export_args["scanpy_coords_name"]][:, 0]
                        sp_data.obs["imagerow"] = sp_data.obsm[export_args["scanpy_coords_name"]][:, 1]

                        if export_args["export_mean"]:
                            p = c2lpl.plot_factor_spatial(
                                adata=sp_data,
                                fact_ind=np.arange(mod.spot_factors_df.shape[1]),
                                fact=mod.spot_factors_df,
                                cluster_names=cluster_plot_names,
                                n_columns=6,
                                trans="identity",
                                sample_name=s,
                                samples_col=train_args["sample_name_col"],
                                obs_x="imagecol",
                                obs_y="imagerow",
                            )
                            p.save(
                                filename=fig_path
                                + "cell_locations_W_mRNA_count_"
                                + str(s)
                                + "."
                                + export_args["plot_extension"]
                            )

                        if export_args["export_q05"]:
                            p = c2lpl.plot_factor_spatial(
                                adata=sp_data,
                                fact_ind=np.arange(mod.spot_factors_q05.shape[1]),
                                fact=mod.spot_factors_q05,
                                cluster_names=cluster_plot_names,
                                n_columns=6,
                                trans="identity",
                                sample_name=s,
                                samples_col=train_args["sample_name_col"],
                                obs_x="imagecol",
                                obs_y="imagerow",
                            )
                            p.save(
                                filename=fig_path
                                + "cell_locations_W_mRNA_count_q05_"
                                + str(s)
                                + "."
                                + export_args["plot_extension"]
                            )

                    if show_locations:
                        print(p)
        except Exception as e:
            print("Some error in plotting with scanpy or `cell2location.plt.plot_factor_spatial()`\n " + repr(e))

    if verbose:
        print("### Done ### - time " + str(np.around((time.time() - start) / 60, 2)) + " min")

    if return_all:
        return res_dict
    else:
        del res_dict
        del mod
        gc.collect()
        return str((time.time() - start) / 60) + " min"
