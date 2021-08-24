# -*- coding: utf-8 -*-
"""Wrapper for the regression model for estimating reference expression signatures of cell types (and other covariates),
accounting for the effects of experiment batch and technology."""

import gc
import os
import pickle

# +
import time
from os import mkdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cell2location.models.reference as models

# from scipy.sparse import issparse


def save_plot(path, filename, extension="png"):
    r"""Save current plot to `path` as `filename` with `extension`"""

    plt.savefig(path + filename + "." + extension)
    # fig.clear()
    # plt.close()


def run_regression(
    sc_data,
    model_name=None,
    verbose=True,
    return_all=True,
    train_args={
        "covariate_col_names": [None],
        "sample_name_col": None,
        "tech_name_col": None,
        "stratify_cv": None,
        "n_epochs": 100,
        "minibatch_size": 1024,
        "learning_rate": 0.01,
        "use_average_as_initial_value": True,
        "use_cuda": True,
        "train_proportion": 0.9,
        "l2_weight": True,  # uses defaults for the model
        "readable_var_name_col": None,
    },
    model_kwargs={},
    posterior_args={},
    export_args={"path": "./results", "save_model": True, "run_name_suffix": ""},
):
    r"""Estimation of cell type reference signatures from scRNA-seq.

    Given cell type annotation for each cell,
    the corresponding reference cell type signatures :math:`g_{f,g}`, which represent the average mRNA count
    of each gene g in each cell type :math:`f={1, .., F}`, can be estimated using a negative binomial regression model,
    which allows for combining data across batches and technologies (see `cell2location.run_regression`).

    This wrapper function performs the following steps: train, evaluate using cross-validation, choose training time that prevents over-fitting,
    evaluate how well the model corrected batch (experiment) and technology effect, save the model object, export results and save diagnostic plots.

    Parameters
    ----------
    sc_data :
        anndata object with single cell / nucleus data
    model_name :
        model class object or None. If none, the default model is selected:
        If `train_args['tech_name_col']` is not provided RegressionNBV2Torch model is used,
        if `train_args['tech_name_col']` is specified RegressionNBV4Torch model is used.

    Returns
        -------
        dict
            dictionary {'mod','sc_data','model_name', 'train_args','posterior_args', 'export_args', 'run_name', 'run_time'}
    """

    # set default parameters

    d_train_args = {
        "covariate_col_names": [None],
        "sample_name_col": None,
        "tech_name_col": None,
        "stratify_cv": None,
        "n_epochs": 100,
        "minibatch_size": 1024,
        "learning_rate": 0.01,
        "minibatch_seed": [41, 56, 345, 12, 6, 3],
        "use_average_as_initial_value": True,
        "use_cuda": True,
        "train_proportion": 0.9,
        "retrain": True,
        "l2_weight": True,  # uses defaults for the model
        "sample_prior": False,
        "readable_var_name_col": None,
        "use_raw": True,
        "mode": "normal",
        "n_type": "restart",
        "n_restarts": 2,
        "checkpoints": None,
    }

    d_posterior_args = {
        "n_samples": 1000,
        "evaluate_stability_align": False,
        "evaluate_stability_transpose": True,
        "mean_field_slot": "init_1",
    }

    d_export_args = {
        "path": "./results",
        "plot_extension": "png",
        "save_model": True,
        "run_name_suffix": "",
        "export_q05": False,
    }

    # replace defaults with parameters supplied
    for k in train_args.keys():
        d_train_args[k] = train_args[k]
    train_args = d_train_args
    for k in posterior_args.keys():
        d_posterior_args[k] = posterior_args[k]
    posterior_args = d_posterior_args
    for k in export_args.keys():
        d_export_args[k] = export_args[k]
    export_args = d_export_args

    # start timing
    start = time.time()

    sc_data = sc_data.copy()

    ####### Preparing data #######

    # extract data as a dense matrix
    if train_args["use_raw"]:
        X_data = sc_data.raw.X.toarray()
    else:
        X_data = sc_data.X.toarray()

    if train_args["sample_name_col"] is None:
        sc_data.obs["sample"] = "sample"
        train_args["sample_name_col"] = "sample"
    model_kwargs["sample_id"] = train_args["sample_name_col"]

    if train_args["readable_var_name_col"] is not None:
        model_kwargs["var_names_read"] = sc_data.var[train_args["readable_var_name_col"]]
    else:
        model_kwargs["var_names_read"] = None

    if train_args["tech_name_col"] is not None:
        model_kwargs["tech_id"] = sc_data.obs[train_args["tech_name_col"]]

    if train_args["stratify_cv"] is not None:
        model_kwargs["stratify_cv"] = sc_data.obs[train_args["stratify_cv"]]

    if train_args["use_cuda"] is not None:
        model_kwargs["use_cuda"] = train_args["use_cuda"]

    fit_kwards = {}
    if train_args["n_epochs"] is not None:
        fit_kwards["n_iter"] = train_args["n_epochs"]
    if train_args["learning_rate"] is not None:
        fit_kwards["learning_rate"] = train_args["learning_rate"]
    if train_args["train_proportion"] is not None:
        fit_kwards["train_proportion"] = train_args["train_proportion"]
    if train_args["n_type"] is not None:
        fit_kwards["n_type"] = train_args["n_type"]
    if train_args["n_restarts"] is not None:
        fit_kwards["n"] = train_args["n_restarts"]
    if train_args["l2_weight"] is not None:
        fit_kwards["l2_weight"] = train_args["l2_weight"]
    if train_args["use_raw"] is True:
        train_args["var_names"] = sc_data.raw.var_names
    else:
        train_args["var_names"] = sc_data.var_names

    train_args["checkpoint_dir"] = f'{export_args["path"]}/checkpoints'
    if train_args["checkpoints"] is not None:
        fit_kwards["checkpoints"] = train_args["checkpoints"]
        fit_kwards["checkpoint_dir"] = train_args["checkpoint_dir"]
        if not os.path.exists(fit_kwards["checkpoint_dir"]):
            mkdir(fit_kwards["checkpoint_dir"])

    # extract pd.DataFrame with covariates
    cell2covar = sc_data.obs[[train_args["sample_name_col"]] + train_args["covariate_col_names"]]

    sample_n = len(np.unique(sc_data.obs[train_args["sample_name_col"]]))
    if sample_n == sc_data.shape[0]:
        ValueError("The number of samples (batches) is equal to the number of observations `adata.n_obs`")

    ####### Creating model #######
    if verbose:
        print("### Creating model ### - time " + str(np.around((time.time() - start) / 60, 2)) + " min")

    ####### Creating model #######
    # choose model when not specified
    if model_name is None:
        if train_args["tech_name_col"] is None:
            Model = models.RegressionGeneBackgroundCoverageTorch
        else:
            Model = models.RegressionGeneBackgroundCoverageGeneTechnologyTorch
    else:  # use supplied class
        Model = model_name

    mod = Model(
        cell2covar=cell2covar,
        X_data=X_data,
        n_iter=train_args["n_epochs"],
        learning_rate=train_args["learning_rate"],
        var_names=train_args["var_names"],
        minibatch_size=train_args["minibatch_size"],
        minibatch_seed=train_args["minibatch_seed"],
        use_average_as_initial_value=train_args["use_average_as_initial_value"],
        verbose=False,
        **model_kwargs,
    )

    ####### Print run name #######
    run_name = (
        str(mod.__class__.__name__)
        + "_"
        + str(mod.n_fact)
        + "covariates_"
        + str(mod.n_obs)
        + "cells_"
        + str(mod.n_var)
        + "genes"
        + export_args["run_name_suffix"]
    )

    print("### Analysis name: " + run_name)  # analysis name is always printed

    # create the export directory
    path = export_args["path"] + run_name + "/"
    if not os.path.exists(path):
        os.makedirs(os.path.abspath(path))

    fig_path = path + "plots/"
    if not os.path.exists(fig_path):
        mkdir(fig_path)

    ####### Sampling prior #######
    if train_args["sample_prior"]:
        raise ValueError("Sampling prior not implemented yet")

    ####### Training model #######
    if verbose:
        print("### Training model to determine n_epochs with CV ###")
    if train_args["mode"] == "normal":
        mod.train(**fit_kwards)
    elif train_args["mode"] == "tracking":
        mod.train(tracking=True, **fit_kwards)
    else:
        raise ValueError("train_args['mode'] can be only 'normal' or 'tracking'")

    ####### Evaluate cross-validation
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 5))

    if train_args["train_proportion"] is None:
        mod.plot_history(0, train_args["n_epochs"], mean_field_slot="init_1", ax=axs[0])
    else:
        mod.plot_validation_history(0, train_args["n_epochs"], mean_field_slot="init_1", ax=axs[0])
    axs[0].get_legend().remove()

    if train_args["train_proportion"] is None:
        mod.plot_history(0, train_args["n_epochs"], mean_field_slot="init_2", ax=axs[1])
    else:
        mod.plot_validation_history(0, train_args["n_epochs"], mean_field_slot="init_2", ax=axs[1])
    axs[1].get_legend().remove()
    axs[1].set_ylabel(None)

    if train_args["train_proportion"] is None:
        mod.plot_history(0, int(np.min((train_args["n_epochs"], 30))), mean_field_slot="init_1", ax=axs[2])
    else:
        mod.plot_validation_history(0, int(np.min((train_args["n_epochs"], 30))), mean_field_slot="init_1", ax=axs[2])
    axs[2].set_ylabel(None)

    plt.tight_layout()
    save_plot(fig_path, filename="training_and_cv_history", extension=export_args["plot_extension"])
    if verbose:
        plt.show()
    plt.close()

    ####### Use cross-validation to select the last epoch before validation loss increased (derivative > 0)
    if train_args["train_proportion"] is not None:

        new_n_epochs = []
        for tr in mod.validation_hist.values():
            deriv = np.gradient(tr, 1)
            new_n_epochs.append(np.max(np.arange(deriv.shape[0])[(deriv < 0)]))
        new_n_epochs = np.min(new_n_epochs) + 1

        ####### Repeat training up until that iteration
        if train_args["retrain"] is True:
            if verbose:
                print("### Re-training model to stop before overfitting ###")
            fit_kwards["n_iter"] = int(new_n_epochs)
            mod.train(**fit_kwards)
            # save the training and validation loss history

            plt.figure(figsize=(5, 5))
            mod.plot_validation_history(0)
            plt.tight_layout()
            save_plot(fig_path, filename="re_training_and_cv_history", extension=export_args["plot_extension"])
            if verbose:
                plt.show()
            plt.close()
        else:
            file_path = f'{train_args["checkpoint_dir"]}/{posterior_args["mean_field_slot"]}_{new_n_epochs}.ckp'
            if file_path in os.listdir(train_args["checkpoint_dir"]):
                if verbose:
                    print(
                        f'### Loading checkpoint before overfitting: {posterior_args["mean_field_slot"]}_{new_n_epochs}.ckp ###'
                    )
                mod.load_checkpoint(file_path)

    ####### Evaluate stability of training #######
    if train_args["n_restarts"] > 1:
        n_plots = train_args["n_restarts"] - 1
        ncol = int(np.min((n_plots, 3)))
        nrow = np.ceil(n_plots / ncol)
        plt.figure(figsize=(5 * nrow, 5 * ncol))
        mod.evaluate_stability(
            n_samples=posterior_args["n_samples"],
            align=posterior_args["evaluate_stability_align"],
            transpose=posterior_args["evaluate_stability_transpose"],
        )
        save_plot(fig_path, filename="evaluate_stability", extension=export_args["plot_extension"])
        if verbose:
            plt.show()
        plt.close()

    ####### Evaluating parameters / sampling posterior #######
    if verbose:
        print(
            f"### Evaluating parameters / sampling posterior ### - time {np.around((time.time() - start) / 60, 2)} min"
        )
    # extract all parameters from parameter store or sample posterior
    mod.sample_posterior(
        node="all",
        n_samples=posterior_args["n_samples"],
        save_samples=False,
        mean_field_slot=posterior_args["mean_field_slot"],
    )

    # evaluate predictive accuracy of the model
    mod.compute_expected()

    # Predictive accuracy
    try:
        mod.plot_posterior_mu_vs_data()
        save_plot(fig_path, filename="data_vs_posterior_mean", extension=export_args["plot_extension"])
        if verbose:
            plt.show()
        plt.close()
    except Exception as e:
        print("Some error in plotting `mod.plot_posterior_mu_vs_data()`\n " + str(e))

    ####### Export summarised posterior & Saving results #######
    if verbose:
        print("### Saving results ###")

    # save covariate effects
    # convert additive sample and covariate effects to pd.DataFrame
    mod.sample2df()
    # annotate single cell data object
    sc_data = mod.annotate_cell_adata(sc_data, use_raw=train_args["use_raw"])

    # save
    mod.covariate_effects.to_csv(path + "covariate_effects.csv")
    mod.sample_effects.to_csv(path + "sample_effects.csv")
    if export_args["export_q05"]:
        mod.covariate_effects_q05.to_csv(path + "covariate_effects_q05.csv")
        mod.sample_effects_q05.to_csv(path + "sample_effects_q05.csv")

    # add posterior of all parameters to `sc_data.uns['regression_mod']`
    mod.fact_filt = None
    sc_data = mod.export2adata(sc_data, slot_name="regression_mod")
    sc_data.uns["regression_mod"]["fact_names"] = list(sc_data.uns["regression_mod"]["fact_names"])
    sc_data.uns["regression_mod"]["var_names"] = list(sc_data.uns["regression_mod"]["var_names"])
    sc_data.uns["regression_mod"]["obs_names"] = list(sc_data.uns["regression_mod"]["obs_names"])

    # save anndata with exported posterior
    sc_data.write(filename=path + "sc.h5ad", compression="gzip")

    # save model object and related annotations
    if export_args["save_model"]:
        # save the model and other objects
        mod.X_data = None
        mod.x_data = None
        mod.mu = None
        res_dict = {
            "mod": mod,
            "model_name": model_name,
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
            "model_name": model_name,
            "train_args": train_args,
            "posterior_args": posterior_args,
            "export_args": export_args,
            "run_name": run_name,
            "run_time": str(np.around((time.time() - start) / 60, 2)) + " min",
        }
        pickle.dump(res_dict, file=open(path + "model_.p", "wb"))

    ####### Plotting #######
    if verbose:
        print("### Plotting results ###")

    # Inferred sample_scaling
    try:
        sc_data.obs["n_counts"] = np.array(sc_data.raw.X.sum(1)).flatten()
        mean_total_count = []
        for s in sc_data.obs[train_args["sample_name_col"]].unique():
            a = sc_data.obs.loc[sc_data.obs[train_args["sample_name_col"]].isin([s]), "n_counts"].mean()
            mean_total_count.append(a)

        mean_total_count = pd.Series(mean_total_count, index=sc_data.obs[train_args["sample_name_col"]].unique())
        # name and order samples the same way as in the model:
        mean_total_count.index = [
            f'means_sample_effect{train_args["sample_name_col"]}_{i}' for i in mean_total_count.index
        ]
        mean_total_count = mean_total_count[mod.sample_effects.columns]

        plt.figure(figsize=(5, 5))
        plt.scatter(np.array(mean_total_count), mod.samples["post_sample_means"]["sample_scaling"].flatten())
        plt.xlabel("Mean total mRNA count per cell")
        plt.ylabel("Inferred sample_scaling")
        save_plot(fig_path, filename="evaluating_sample_scaling", extension=export_args["plot_extension"])
        if verbose:
            plt.show()
        plt.close()
    except Exception as e:
        print("Some error in plotting inferred sample_scaling\n " + str(e))

    # Inferred over-dispersion
    try:
        inferred_shape = 1 / (mod.samples["post_sample_means"]["gene_E"] * mod.samples["post_sample_means"]["gene_E"])
        plt.hist(inferred_shape.flatten(), bins=50)
        plt.xlabel("Inferred over-dispersion (shape of Gamma distribution")
        plt.tight_layout()
        save_plot(fig_path, filename="inferred_over_dispersion", extension=export_args["plot_extension"])
        if verbose:
            plt.show()
        plt.close()
    except Exception as e:
        print("Some error in plotting `Inferred over-dispersion`\n " + str(e))

    if verbose:
        print("### Done ### - time " + res_dict["run_time"])

    if return_all:
        res_dict["mod"] = mod
        return res_dict, sc_data
    else:
        del res_dict
        del mod
        gc.collect()
        return str((time.time() - start) / 60) + " min"
