# -*- coding: utf-8 -*-
r"""Base Pymc3 model class for all models in pymc3"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
from pymc3.variational.callbacks import CheckParametersConvergence
from tqdm.auto import tqdm

from cell2location.models.base.base_model import BaseModel


# base model class - defining shared methods but not the model itself
class Pymc3Model(BaseModel):
    r"""This class provides functions to train PyMC3 models and sample their parameters.
    A model must have a main X_data input and can have arbitrary self.extra_data inputs.

    Parameters
    ----------
    X_data :
        Numpy array of gene expression (cols) in spatial locations (rows)
    all other:
        the rest are arguments for parent class BaseModel

    """

    def __init__(
        self,
        X_data: np.ndarray,
        n_fact: int = 10,
        data_type: str = "float32",
        n_iter: int = 200000,
        learning_rate=0.001,
        total_grad_norm_constraint=200,
        verbose=True,
        var_names=None,
        var_names_read=None,
        obs_names=None,
        fact_names=None,
        sample_id=None,
    ):

        ############# Initialise parameters ################
        super().__init__(
            X_data,
            n_fact,
            data_type,
            n_iter,
            learning_rate,
            total_grad_norm_constraint,
            verbose,
            var_names,
            var_names_read,
            obs_names,
            fact_names,
            sample_id,
        )

        # Create dictionaries storing results
        self.advi = {}
        self.mean_field = {}
        self.samples = {}
        self.node_samples = {}
        self.n_type = "restart"  # default

        # Pass data to theano
        self.x_data = theano.shared(X_data.astype(self.data_type))

    def sample_prior(self, samples=10):
        r"""Take samples from the prior, see `pymc3.sample_prior_predictive` for details

        Parameters
        ----------
        samples :
             (Default value = 10)

        Returns
        -------
        dict
            self.prior_trace dictionary with an element for each parameter of the mode
        """
        # Take samples from the prior
        with self.model:
            self.prior_trace = pm.sample_prior_predictive(samples=samples)

    def fit_advi(self, n=3, method="advi", n_type="restart"):
        r"""Find posterior using ADVI (maximising likehood of the data and
        minimising KL-divergence of posterior to prior)

        Parameters
        ----------
        n :
            number of independent initialisations (Default value = 3)
        method :
            advi', to allow for potential use of SVGD, MCMC, custom (currently only ADVI implemented). (Default value = 'advi')
        n_type :
            type of repeated initialisation:

            * **'restart'** to pick different initial value,
            * **'cv'** for molecular cross-validation - splits counts into n datasets, for now, only n=2 is implemented
            * **'bootstrap'** for fitting the model to multiple downsampled datasets.
              Run `mod.bootstrap_data()` to generate variants of data (Default value = 'restart')

        Returns
        -------
        dict
            self.mean_field dictionary with MeanField pymc3 objects for each initialisation.

        """

        if not np.isin(n_type, ["restart", "cv", "bootstrap"]):
            raise ValueError("n_type should be one of ['restart', 'cv', 'bootstrap']")

        self.n_type = n_type

        if np.isin(n_type, ["bootstrap"]):
            if self.X_data_sample is None:
                self.bootstrap_data(n=n)
        elif np.isin(n_type, ["cv"]):
            self.generate_cv_data(n=n)  # cv data added to self.X_data_sample

        init_names = ["init_" + str(i + 1) for i in np.arange(n)]

        with self.model:

            for i, name in enumerate(init_names):

                # when type is molecular cross-validation or bootstrap,
                # replace self.x_data tensor with new data
                if np.isin(n_type, ["cv", "bootstrap"]):
                    more_replacements = {self.x_data: self.X_data_sample[i].astype(self.data_type)}
                else:
                    more_replacements = {}

                # train the model
                self.mean_field[name] = pm.fit(
                    self.n_iter,
                    method="advi",
                    callbacks=[CheckParametersConvergence()],
                    obj_optimizer=pm.adam(learning_rate=self.learning_rate),
                    total_grad_norm_constraint=self.total_grad_norm_constraint,
                    more_replacements=more_replacements,
                )

                # plot training history
                if self.verbose:
                    plt.plot(np.log10(self.mean_field[name].hist[15000:]))

    def train(
        self,
        n=3,
        method="advi",
        n_type="restart",
        n_iter=None,
        learning_rate=None,
        reducing_lr=False,
        progressbar=True,
        scale_cost_to_minibatch=True,
    ):
        """Find posterior using pm.ADVI() method directly (allows continuing training through `refine` method.
        (maximising likelihood of the data and minimising KL-divergence of posterior to prior - ELBO loss)

        Parameters
        ----------
        n :
            number of independent initialisations (Default value = 3)
        method :
            advi', to allow for potential use of SVGD, MCMC, custom (currently only ADVI implemented). (Default value = 'advi')
        n_type :
            type of repeated initialisation:

            * **'restart'** to pick different initial value,
            * **'cv'** for molecular cross-validation - splits counts into n datasets, for now, only n=2 is implemented
            * **'bootstrap'** for fitting the model to multiple downsampled datasets.
              Run `mod.bootstrap_data()` to generate variants of data (Default value = 'restart')

        n_iter :
            number of iterations, supersedes self.n_iter specified when creating model instance. (Default value = None)
        learning_rate :
            learning rate, supersedes self.learning_rate specified when creating model instance. (Default value = None)
        reducing_lr :
            boolean, use decaying learning rate? (Default value = False)
        progressbar :
            boolean, show progress bar? (Default value = True)
        scale_cost_to_minibatch :
            when using training in minibatches, scale cost function appropriately?
            See discussion https://discourse.pymc.io/t/effects-of-scale-cost-to-minibatch/1429 to understand the effects. (Default value = True)

        Returns
        -------
        None
            self.mean_field dictionary with MeanField pymc3 objects,
            and self.advi dictionary with ADVI objects for each initialisation.

        """

        self.n_type = n_type
        self.scale_cost_to_minibatch = scale_cost_to_minibatch

        if n_iter is None:
            n_iter = self.n_iter

        if learning_rate is None:
            learning_rate = self.learning_rate

        ### Initialise optimiser ###
        if reducing_lr:
            # initialise the function for adaptive learning rate
            s = theano.shared(np.array(learning_rate).astype(self.data_type))

            def reduce_rate(a, h, i):
                s.set_value(np.array(learning_rate / ((i / self.n_obs) + 1) ** 0.7).astype(self.data_type))

            optimiser = pm.adam(learning_rate=s)
            callbacks = [reduce_rate, CheckParametersConvergence()]
        else:
            optimiser = pm.adam(learning_rate=learning_rate)
            callbacks = [CheckParametersConvergence()]

        if np.isin(n_type, ["bootstrap"]):
            if self.X_data_sample is None:
                self.bootstrap_data(n=n)
        elif np.isin(n_type, ["cv"]):
            self.generate_cv_data()  # cv data added to self.X_data_sample

        init_names = ["init_" + str(i + 1) for i in np.arange(n)]

        for i, name in enumerate(init_names):

            with self.model:

                self.advi[name] = pm.ADVI()

            # when type is molecular cross-validation or bootstrap,
            # replace self.x_data tensor with new data
            if np.isin(n_type, ["cv", "bootstrap"]):

                # defining minibatch
                if self.minibatch_size is not None:
                    # minibatch main data - expression matrix
                    self.x_data_minibatch = pm.Minibatch(
                        self.X_data_sample[i].astype(self.data_type),
                        batch_size=[self.minibatch_size, None],
                        random_seed=self.minibatch_seed[i],
                    )
                    more_replacements = {self.x_data: self.x_data_minibatch}

                    # if any other data inputs should be minibatched add them too
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = pm.Minibatch(
                                self.extra_data[k].astype(self.data_type),
                                batch_size=[self.minibatch_size, None],
                                random_seed=self.minibatch_seed[i],
                            )

                # or using all data
                else:
                    more_replacements = {self.x_data: self.X_data_sample[i].astype(self.data_type)}
                    # if any other data inputs should be added
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = self.extra_data[k].astype(self.data_type)

            else:

                # defining minibatch
                if self.minibatch_size is not None:
                    # minibatch main data - expression matrix
                    self.x_data_minibatch = pm.Minibatch(
                        self.X_data.astype(self.data_type),
                        batch_size=[self.minibatch_size, None],
                        random_seed=self.minibatch_seed[i],
                    )
                    more_replacements = {self.x_data: self.x_data_minibatch}

                    # if any other data inputs should be minibatched add them too
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = pm.Minibatch(
                                self.extra_data[k].astype(self.data_type),
                                batch_size=[self.minibatch_size, None],
                                random_seed=self.minibatch_seed[i],
                            )

                else:
                    more_replacements = {}

            self.advi[name].scale_cost_to_minibatch = scale_cost_to_minibatch

            # train the model
            self.mean_field[name] = self.advi[name].fit(
                n_iter,
                callbacks=callbacks,
                obj_optimizer=optimiser,
                total_grad_norm_constraint=self.total_grad_norm_constraint,
                progressbar=progressbar,
                more_replacements=more_replacements,
            )

            # plot training history
            if self.verbose:
                print(plt.plot(np.log10(self.mean_field[name].hist[15000:])))

    def fit_advi_refine(self, n_iter=10000, learning_rate=None, progressbar=True, reducing_lr=False):
        """Refine posterior using ADVI - continue training after `.fit_advi_iterative()`

        Parameters
        ----------
        n_iter :
            number of additional iterations (Default value = 10000)
        learning_rate :
            same as in `.fit_advi_iterative()` (Default value = None)
        progressbar :
            same as in `.fit_advi_iterative()` (Default value = True)
        reducing_lr :
            same as in `.fit_advi_iterative()` (Default value = False)

        Returns
        -------
        dict
            update the self.mean_field dictionary with MeanField pymc3 objects.

        """

        self.n_iter = self.n_iter + n_iter

        if learning_rate is None:
            learning_rate = self.learning_rate

        ### Initialise optimiser ###
        if reducing_lr:
            # initialise the function for adaptive learning rate
            s = theano.shared(np.array(learning_rate).astype(self.data_type))

            def reduce_rate(a, h, i):
                s.set_value(np.array(learning_rate / ((i / self.n_obs) + 1) ** 0.7).astype(self.data_type))

            optimiser = pm.adam(learning_rate=s)
            callbacks = [reduce_rate, CheckParametersConvergence()]
        else:
            optimiser = pm.adam(learning_rate=learning_rate)
            callbacks = [CheckParametersConvergence()]

        for i, name in enumerate(self.advi.keys()):

            # when type is molecular cross-validation or bootstrap,
            # replace self.x_data tensor with new data
            if np.isin(self.n_type, ["cv", "bootstrap"]):

                # defining minibatch
                if self.minibatch_size is not None:
                    # minibatch main data - expression matrix
                    self.x_data_minibatch = pm.Minibatch(
                        self.X_data_sample[i].astype(self.data_type),
                        batch_size=[self.minibatch_size, None],
                        random_seed=self.minibatch_seed[i],
                    )
                    more_replacements = {self.x_data: self.x_data_minibatch}

                    # if any other data inputs should be minibatched add them too
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = pm.Minibatch(
                                self.extra_data[k].astype(self.data_type),
                                batch_size=[self.minibatch_size, None],
                                random_seed=self.minibatch_seed[i],
                            )

                # or using all data
                else:
                    more_replacements = {self.x_data: self.X_data_sample[i].astype(self.data_type)}
                    # if any other data inputs should be added
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = self.extra_data[k].astype(self.data_type)

            else:
                # defining minibatch
                if self.minibatch_size is not None:
                    # minibatch main data - expression matrix
                    self.x_data_minibatch = pm.Minibatch(
                        self.X_data.astype(self.data_type),
                        batch_size=[self.minibatch_size, None],
                        random_seed=self.minibatch_seed[i],
                    )
                    more_replacements = {self.x_data: self.x_data_minibatch}

                    # if any other data inputs should be minibatched add them too
                    if self.extra_data is not None:
                        # for each parameter in the dictionary add it to more_replacements
                        for k in self.extra_data.keys():
                            more_replacements[self.extra_data_tt[k]] = pm.Minibatch(
                                self.extra_data[k].astype(self.data_type),
                                batch_size=[self.minibatch_size, None],
                                random_seed=self.minibatch_seed[i],
                            )

                else:
                    more_replacements = {}

            with self.model:
                # train for more iterations & export trained model by overwriting the initial mean field object
                self.mean_field[name] = self.advi[name].fit(
                    n_iter,
                    callbacks=callbacks,
                    obj_optimizer=optimiser,
                    total_grad_norm_constraint=self.total_grad_norm_constraint,
                    progressbar=progressbar,
                    more_replacements=more_replacements,
                )

                if self.verbose:
                    print(plt.plot(np.log10(self.mean_field[name].hist[15000:])))

    def plot_history(self, iter_start=0, iter_end=-1):
        """Plot loss function (ELBO) across training history

        Parameters
        ----------
        iter_start :
            omit initial iterations from the plot (Default value = 0)
        iter_end :
            omit last iterations from the plot (Default value = -1)

        """
        for i in self.mean_field.keys():
            print(plt.plot(np.log10(self.mean_field[i].hist[iter_start:iter_end])))

    def b_evaluate_stability(self, node, fact_filt=None, n_samples=1000, align=True, batch_size: int = 40):
        """Evaluate stability of posterior samples between training initialisations
        (takes samples and correlates the values of factors between training initialisations)

        Parameters
        ----------
        node :
            which pymc3 node (model parameter) to sample? Factors should be in columns.
        n_samples :
            the number of samples. (Default value = 1000)
        align :
            boolean, match factors between training restarts using linear_sum_assignment? (Default value = True)
        batch_size :
            int, the number of samples to generate per batch (conserves GPU memory allocation).

        Returns
        -------
        dict
            self.samples[node_name+_stab] dictionary with an element for each training initialisation.

        """

        theano.config.compute_test_value = "ignore"

        self.n_samples = n_samples
        n_batches = int(n_samples / batch_size)

        node_name = node.name
        self.samples[node_name + "_stab"] = {}

        for i in self.mean_field.keys():
            # Sample a single node
            post_node = self.mean_field[i].sample_node(node, size=batch_size).eval()

            for b in tqdm(range(n_batches - 1)):
                # sample remaining batches
                post_node_1 = self.mean_field[i].sample_node(node, size=batch_size).eval()
                # concatenate batches
                post_node = np.concatenate((post_node, post_node_1), axis=0)

            if fact_filt is not None:
                post_node = post_node[:, :, fact_filt]

            self.samples[node_name + "_stab"][i] = post_node.mean(0)

        n_plots = len(self.samples[node_name + "_stab"].keys()) - 1
        for i in range(n_plots):
            ncol = int(np.min((n_plots, 3)))
            nrow = np.ceil(n_plots / ncol)
            plt.subplot(nrow, ncol, i + 1)
            plt.subplot(np.ceil(n_plots / ncol), ncol, i + 1)
            self.align_plot_stability(
                self.samples[node_name + "_stab"]["init_" + str(1)],
                self.samples[node_name + "_stab"]["init_" + str(i + 2)],
                str(1),
                str(i + 2),
                align=align,
            )

    def sample_posterior(
        self,
        node="all",
        n_samples=1000,
        save_samples=False,
        return_samples=True,
        mean_field_slot="init_1",
        batch_size: int = 40,
    ):
        """Sample posterior distribution of all parameters or single parameter

        Parameters
        ----------
        node :
            pymc3 node to sample (e.g. default "all", self.spot_factors)
        n_samples :
            number of posterior samples to generate (1000 is recommended, reduce if you get GPU memory error) (Default value = 1000)
        save_samples :
            save all samples, not just the mean, 5% and 95% quantile, SD. (Default value = False)
        return_samples :
            return summarised samples (mean, etc) in addition to saving them in `self.samples`? (Default value = True)
        mean_field_slot :
            string, which training initialisation (mean_field slot) to sample? 'init_1' by default
        batch_size :
            int, the number of samples to generate per batch (conserves GPU memory allocation).

        Returns
        -------
        dict
            dictionary `self.samples` (mean, 5% quantile, SD, optionally all samples) with dictionaries
            with numpy arrays for each parameter.
            Plus an optional dictionary in `self.samples` with all samples of parameters
            as numpy arrays of shape ``(n_samples, ...)``

        """

        theano.config.compute_test_value = "ignore"
        n_batches = int(n_samples / batch_size)

        if node == "all":

            # Sample all parameters - might use a lot of GPU memory - so done in minibatches
            post_samples = self.mean_field[mean_field_slot].sample(batch_size)

            for i in tqdm(range(n_batches - 1)):
                # sample remaining batches
                post_samples_1 = self.mean_field[mean_field_slot].sample(batch_size)

                try:
                    variable_names = post_samples_1.keys()
                except Exception:
                    variable_names = post_samples_1.varnames

                # concatenate batches
                post_samples = {k: np.concatenate((post_samples[k], post_samples_1[k]), axis=0) for k in variable_names}

            try:
                variable_names = post_samples.keys()
            except Exception:
                variable_names = post_samples.varnames

            self.samples["post_sample_means"] = {v: post_samples[v].mean(axis=0) for v in variable_names}
            self.samples["post_sample_q05"] = {v: np.quantile(post_samples[v], 0.05, axis=0) for v in variable_names}
            self.samples["post_sample_q95"] = {v: np.quantile(post_samples[v], 0.95, axis=0) for v in variable_names}
            self.samples["post_sample_sds"] = {v: post_samples[v].std(axis=0) for v in variable_names}

            if save_samples:
                self.samples["post_samples"] = post_samples

        else:
            # Sample a single node
            post_node = self.mean_field[mean_field_slot].sample_node(node, size=batch_size).eval()

            for i in tqdm(range(n_batches - 1)):
                # sample remaining batches
                post_node_1 = self.mean_field[mean_field_slot].sample_node(node, size=batch_size).eval()
                # concatenate batches
                post_node = np.concatenate((post_node, post_node_1), axis=0)

            post_node_mean = post_node.mean(0)
            post_node_q05 = np.quantile(post_node, 0.05, axis=0)
            post_node_q95 = np.quantile(post_node, 0.95, axis=0)
            post_node_sds = post_node.std(0)

            # extract the name of the node and save to samples dictionary
            node_name = node.name
            self.node_samples[str(node_name) + "_mean"] = post_node_mean
            self.node_samples[str(node_name) + "_q05"] = post_node_q05
            self.node_samples[str(node_name) + "_q95"] = post_node_q95
            self.node_samples[str(node_name) + "_sds"] = post_node_sds

            if save_samples:
                self.node_samples[str(node_name) + "_post_samples"] = post_node

        if return_samples:
            return self.samples

    def factor_expressed_plot(
        self,
        shape_cut=4,
        rate_cut=15,
        sample_type="post_sample_means",
        shape="cell_fact_mu_hyp",
        rate="cell_fact_sd_hyp",
        x_lab="Cell factor, Gamma shape",
        y_lab="Cell factor, Gamma rate",
    ):
        """Not used for Location models. Ignore.
        Identify which factors are expressed and filter them

        Parameters
        ----------
        shape_cut :
            cutoff on X-axis (Default value = 4)
        rate_cut :
            cutoff on Y-axis (Default value = 15)
        sample_type :
            which posterior summary type to use. (Default value = 'post_sample_means')
        shape :
            name for X-axis parameter (Default value = 'cell_fact_mu_hyp')
        rate :
            name for Y-axis parameter (Default value = 'cell_fact_sd_hyp')
        x_lab :
            X-axis label (Default value = 'Cell factor)
        y_lab :
            Y-axis label (Default value = 'Cell factor)
        Gamma shape' :

        Gamma rate' :


        Returns
        -------
        dict
            a dictionary with extracted parameters and cutoffs {'shape': shape, 'rate': rate,
            'shape_cut': shape_cut, 'rate_cut': rate_cut}

        """

        # Expression shape and rate across cells
        shape = self.samples[sample_type][shape]
        rate = self.samples[sample_type][rate]
        plt.scatter(shape, rate)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.vlines(shape_cut, 0, rate_cut)
        plt.hlines(rate_cut, 0, shape_cut)
        plt.text(shape_cut - 0.5 * shape_cut, rate_cut - 0.5 * rate_cut, "expressed")
        plt.text(shape_cut + 0.1 * shape_cut, rate_cut + 0.1 * rate_cut, "not expressed")

        return {"shape": shape, "rate": rate, "shape_cut": shape_cut, "rate_cut": rate_cut}

    def track_parameters(
        self,
        n=1,
        every=1000,
        n_samples=100,
        n_type="restart",
        df_node_name1="nUMI_factors",
        df_node_df_name1="spot_factors_df",
        df_prior_node_name1="spot_fact_mu_hyp",
        df_prior_node_name2="spot_fact_sd_hyp",
        mu_node_name="mu",
        data_node="X_data",
        extra_df_parameters=("spot_add"),
    ):
        """Track posterior distribution of all parameters during training.
        This is done by taking `n_samples` posterior samples `every` 1000 iterations. If `n_type` is 'cv',
        or cross-validation, this function also computes RMSE on training and validation data (molecular cross-validation)

        Parameters
        ----------
        every :
            save posterior `every` iterations (Default value = 1000)
        n_samples :
            number of posterior samples to generate (Default value = 100)
        n :
            number of independent training initialisations (Default value = 1)
        n_type :
            see .fit_advi_iterative() method (Default value = 'restart')
        df_node_name1 :
            which node to convert to posterior with `self.sample2df`? (Default value = 'nUMI_factors')
        df_node_df_name1 :
            names of the object where `self.sample2df` stores pd.DataFrame for `df_node_name1` (Default value = 'spot_factors_df')
        df_prior_node_name1 :
            first hierarchical prior for `df_node_name1`,
            set to None to not paste these values to column names (Default value = 'spot_fact_mu_hyp')
        df_prior_node_name2 :
            second hierarchical prior for `df_node_name1`,
            set to None to not paste these values to column names (Default value = 'spot_fact_sd_hyp')
        mu_node_name :
            name of the object slot containing expected value.
            This is used to calculate RMSE on training and validation data (Default value = 'mu')
        data_node :
            name of the object slot containing data
            This is used to calculate RMSE on training and validation data (Default value = 'X_data')
        extra_df_parameters :
            a tuple with names of additional parameters
            that can be added to `df_node_df_name1` pd.DataFrame. (Default value = ('spot_add'))

        Returns
        -------
        None
            adds self.tracking dictionary with one element for each `n` containing:

            * **'samples'**: a list of length `self.n_iter / every` with every entry being
              a dictionary of parameters (one entry per parameter)
            * **'samples_df'**: a list of length `self.n_iter / every` with every entry being
              a pd.DataFrame for cell factors
            * **'rmse'**: a list of length `self.n_iter / every` with every entry being
              a dictionary ('rmse_genes', 'rmse_cells', 'rmse_total'). When using cross-validation data,
              RMSE for validation data are also added ('rmse_genes_cv', 'rmse_cells_cv', 'rmse_total_cv').

        """

        self.tracking_n_steps = np.ceil(self.n_iter / every)
        self.tracking_every = every

        if n_type == "cv":
            n = 2

        # initialise tracking dictionary
        init_names = ["init_" + str(i + 1) for i in np.arange(n)]
        self.tracking = {n_init: {"samples": [], "samples_df": [], "rmse": []} for n_init in init_names}

        #### train for self.n_iter in n_steps
        for i in tqdm(range(int(self.tracking_n_steps + 1))):

            if i == 0:
                # initialise and do one step
                self.train(n=n, method="advi", n_type=n_type, n_iter=1, progressbar=False)
            else:
                # Refine
                self.fit_advi_refine(every, progressbar=False)

            #### for each training initialisation generate and record samples
            for n_init in init_names:

                #### sample posterior ####
                self.sample_posterior(node="all", n_samples=n_samples, save_samples=False, mean_field_slot=str(n_init))

                # save samples of all parameters
                self.tracking[n_init]["samples"] = self.tracking[n_init]["samples"] + [self.samples.copy()]

                #### compute root mean squared error for genes and cells ####
                self.compute_expected()
                mu = getattr(self, mu_node_name)

                if n_type == "restart":
                    data = getattr(self, data_node)

                    rmse = (mu - data) * (mu - data)
                    rmse_dict = {
                        "rmse_genes": np.sqrt(rmse.mean(0)),
                        "rmse_cells": np.sqrt(rmse.mean(1)),
                        "rmse_total": np.sqrt(rmse.mean()),
                    }
                #### in cross-validation:
                #### compute root mean squared error against the other dataset
                elif n_type == "cv":

                    data = self.X_data_sample[init_names == n_init]
                    data_cv = self.X_data_sample[init_names != n_init]

                    rmse = (mu - data) * (mu - data)
                    rmse_cv = (mu - data_cv) * (mu - data_cv)
                    rmse_dict = {
                        "rmse_genes": np.sqrt(rmse.mean(0)),
                        "rmse_cells": np.sqrt(rmse.mean(1)),
                        "rmse_total": np.sqrt(rmse.mean()),
                        "rmse_genes_cv": np.sqrt(rmse_cv.mean(0)),
                        "rmse_cells_cv": np.sqrt(rmse_cv.mean(1)),
                        "rmse_total_cv": np.sqrt(rmse_cv.mean()),
                    }

                # save RMSE measurement
                self.tracking[n_init]["rmse"] = self.tracking[n_init]["rmse"] + [rmse_dict]

                #### extract a data frame for spots / cells ####
                self.sample2df(node_name=df_node_name1)
                factors_df = getattr(self, df_node_df_name1)

                # add hyperprior to names
                if df_prior_node_name1 is not None:
                    df_prior_1 = self.samples["post_sample_means"][df_prior_node_name1]
                    factors_df.columns = (
                        factors_df.columns + "|" + pd.Series(np.char.mod("%d", np.around(df_prior_1.flatten(), 1)))
                    )
                if df_prior_node_name2 is not None:
                    df_prior_2 = self.samples["post_sample_means"][df_prior_node_name2]
                    factors_df.columns = (
                        factors_df.columns + "|" + pd.Series(np.char.mod("%d", np.around(df_prior_2.flatten(), 1)))
                    )

                #### add the reconstruction error for spots/cells
                factors_df["rmse_cells"] = rmse_dict["rmse_cells"]
                if n_type == "cv":
                    factors_df["rmse_cells_cv"] = rmse_dict["rmse_cells_cv"]

                #### add extra parameters that also describe spots/cells
                if extra_df_parameters is not None:

                    for par in extra_df_parameters:
                        factors_df[par + "_means"] = self.samples["post_sample_means"][par].flatten()
                        factors_df[par + "_q05"] = self.samples["post_sample_q05"][par].flatten()
                        factors_df[par + "_q95"] = self.samples["post_sample_q95"][par].flatten()

                #### save the DataFrame for spots/cells ####
                self.tracking[n_init]["samples_df"] = self.tracking[n_init]["samples_df"] + [factors_df.copy()]

    def sample_posterior_bootstrap(self, n_samples=100, save_samples=True):
        """Sample posterior using training on bootstrapped data.
        This method could be give a better estimate of posterior variance than ADVI.
        SD of the posterior is calculated based on fits to bootstrapped data rather
        than samples from each posterior.

        Parameters
        ----------
        n_samples :
            the same as `.sample_posterior()` (Default value = 100)
        save_samples :
            In this case, samples are values based on each bootstrapped dataset. (Default value = True)

        """

        post_samples = self.sample_posterior(
            node="all", n_samples=n_samples, save_samples=False, mean_field_slot=list(self.mean_field.keys())[0]
        )
        post_samples = post_samples["post_sample_means"]

        try:
            variable_names = post_samples.keys()
        except Exception:
            variable_names = post_samples.varnames

        post_samples = {v: np.array([post_samples[v] for i in self.mean_field.keys()]) for v in variable_names}

        for i, n in enumerate(list(self.mean_field.keys())[1:]):

            sample = self.sample_posterior(node="all", n_samples=n_samples, save_samples=False, mean_field_slot=n)
            for k in variable_names:
                post_samples[k][i + 1] = sample["post_sample_means"][k]

        self.samples = {
            "post_sample_means": {v: post_samples[v].mean(axis=0) for v in variable_names},
            "post_sample_q05": {v: np.quantile(post_samples[v], 0.05, axis=0) for v in variable_names},
            "post_sample_q95": {v: np.quantile(post_samples[v], 0.95, axis=0) for v in variable_names},
            "post_sample_sds": {v: post_samples[v].std(axis=0) for v in variable_names},
        }

        if save_samples:
            self.samples["post_samples"] = post_samples

        return self.samples
