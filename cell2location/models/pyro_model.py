# -*- coding: utf-8 -*-
"""Base pyro model class"""

import os
from collections import defaultdict, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyro import poutine
from tqdm.auto import tqdm

import pyro
import torch
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDelta, AutoGuideList
from cell2location.distributions.AutoNormal import AutoNormal
from pyro.infer.autoguide import init_to_mean
import pyro.optim as optim
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from cell2location.models.base_model import BaseModel


def flatten_iterable(iterable):
    flattened_list = []
    for i in iterable:
        if isinstance(i, Iterable):
            flattened_list += flatten_iterable(i)
        else:
            flattened_list.append(i)
    return flattened_list


class MiniBatchDataset(Dataset):

    def __init__(self, x_data, extra_data, return_idx=False):
        self.x_data = x_data
        self.extra_data = extra_data
        self.return_idx = return_idx

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.return_idx:
            return self.x_data[idx], {**{'idx': idx}, **{k: v[idx] for k, v in self.extra_data.items()}}
        return self.x_data[idx], {k: v[idx] for k, v in self.extra_data.items()}


# base model class - defining shared methods but not the model itself
class PyroModel(BaseModel):
    r"""Base class for pyro models.
    :param n_fact: number of factors
    :param X_data: Numpy array of gene expression (cols) in spatial locations (rows)
    :param learning_rate: ADAM learning rate for optimising Variational inference objective
    :param n_iter: number of training iterations
    :param total_grad_norm_constraint: gradient constraints in optimisation
    """

    def __init__(
            self,
            X_data: np.ndarray,
            n_fact: int = 10,
            data_type: str = 'float32',
            n_iter: int = 200000,
            learning_rate=0.001,
            total_grad_norm_constraint=200,
            use_cuda=True,
            verbose=True,
            var_names=None, var_names_read=None,
            obs_names=None, fact_names=None, sample_id=None,
            minibatch_size=None,
            minibatch_seed=42,
            point_estim=[],
            custom_guides={}
    ):

        ############# Initialise parameters ################
        super().__init__(X_data, n_fact,
                         data_type, n_iter,
                         learning_rate, total_grad_norm_constraint,
                         verbose, var_names, var_names_read,
                         obs_names, fact_names, sample_id)

        self.extra_data = {}
        self.init_vals = {}
        self.minibatch_size = minibatch_size
        self.minibatch_seed = minibatch_seed
        self.MiniBatchDataset = MiniBatchDataset
        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.point_estim = point_estim
        self.custom_guides = custom_guides
        self.guide_type = 'AutoNormal'

        if self.use_cuda:
            if data_type == 'float32':
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
            elif data_type == 'float16':
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
            elif data_type == 'float64':
                torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            else:
                raise ValueError('Only 32, 16 and 64-bit tensors can be used (data_type)')
        else:
            if data_type == 'float32':
                torch.set_default_tensor_type(torch.FloatTensor)
            elif data_type == 'float16':
                torch.set_default_tensor_type(torch.HalfTensor)
            elif data_type == 'float64':
                torch.set_default_tensor_type(torch.DoubleTensor)
            else:
                raise ValueError('Only 32, 16 and 64-bit tensors can be used (data_type)')

    def sample_prior(self):
        r""" Take one sample from the prior 
        :return: self.prior_trace dictionary with an element for each parameter of the model. 
        """
        print(".sample_prior() not implemented yet")

    def fit_advi_iterative_simple(self, n: int = 3, method='advi', n_type='restart',
                                  n_iter=None, learning_rate=None, progressbar=True, ):
        r""" Find posterior using ADVI (deprecated)
        (maximising likehood of the data and minimising KL-divergence of posterior to prior)
        :param n: number of independent initialisations
        :param method: which approximation of the posterior (guide) to use?.
            * ``'advi'`` - Univariate normal approximation (pyro.infer.autoguide.AutoDiagonalNormal)
            * ``'custom'`` - Custom guide using conjugate posteriors
        :return: self.svi dictionary with svi pyro objects for each n, and sefl.elbo dictionary storing training history. 
        """

        # Pass data to pyro / pytorch
        self.x_data = torch.tensor(self.X_data.astype(self.data_type))  # .double()

        # initialise parameter store
        self.svi = {}
        self.hist = {}
        self.guide_i = {}
        self.samples = {}
        self.node_samples = {}

        self.n_type = n_type

        if n_iter is None:
            n_iter = self.n_iter

        if learning_rate is None:
            learning_rate = self.learning_rate

        if np.isin(n_type, ['bootstrap']):
            if self.X_data_sample is None:
                self.bootstrap_data(n=n)
        elif np.isin(n_type, ['cv']):
            self.generate_cv_data()  # cv data added to self.X_data_sample

        init_names = ['init_' + str(i + 1) for i in np.arange(n)]

        for i, name in enumerate(init_names):

            # initialise Variational distributiion = guide
            if method is 'advi':
                self.guide_i[name] = AutoGuideList(self.model)
                self.guide_i[name].append(
                    AutoNormal(poutine.block(self.model, expose_all=True, hide_all=False, hide=self.point_estim),
                               init_loc_fn=init_to_mean))
                self.guide_i[name].append(
                    AutoDelta(poutine.block(self.model, hide_all=True, expose=self.point_estim)))
            elif method is 'custom':
                self.guide_i[name] = self.guide

            # pick dataset depending on the training mode and move to GPU
            if np.isin(n_type, ['cv', 'bootstrap']):
                self.x_data = torch.tensor(self.X_data_sample[i].astype(self.data_type))
            else:
                self.x_data = torch.tensor(self.X_data.astype(self.data_type))

            if self.use_cuda:
                # move tensors and modules to CUDA
                self.x_data = self.x_data.cuda()

            pyro.clear_param_store()

            self.guide_i[name](self.x_data)

            # initialise SVI inference method
            self.svi[name] = SVI(self.model, self.guide_i[name],
                                 optim.ClippedAdam({'lr': learning_rate,
                                                    # limit the gradient step from becoming too large
                                                    'clip_norm': self.total_grad_norm_constraint}),
                                 loss=JitTrace_ELBO())

            # record ELBO Loss history here
            self.hist[name] = []

            # train for n_iter
            it_iterator = tqdm(range(n_iter))
            for it in it_iterator:

                hist = self.svi[name].step(self.x_data)
                it_iterator.set_description('ELBO Loss: ' + str(np.round(hist, 3)))
                self.hist[name].append(hist)

                # if it % 50 == 0 & self.verbose:
                # logging.info("Elbo loss: {}".format(hist))
                if it % 500 == 0:
                    torch.cuda.empty_cache()

    def set_initial_values(self):
        r"""Method for setting initial values on covariate effect (gene_factors parameter)
        :return: nothing
        """
        if self.guide_type == 'AutoGuideList':
            def prefix(i):
                return f'AutoGuideList.{i}.'
        elif self.guide_type == 'AutoNormal':
            def prefix(i):
                return f'AutoNormal.'

        for k in list(self.init_vals.keys()):

            if k in self.point_estim:
                pyro.param(f'{prefix(1)}{k}',
                           torch.Tensor(self.init_vals[k][1](self.init_vals[k][0])))
            else:
                pyro.param(f'{prefix(0)}locs.{k}',
                           torch.Tensor(self.init_vals[k][1](self.init_vals[k][0])))

    def fit_advi_iterative(self, n=3, method='advi', n_type='restart',
                           n_iter=None, learning_rate=None,
                           progressbar=True, num_workers=2,
                           train_proportion=None, stratify_cv=None,
                           l2_weight=False, sample_scaling_weight=0.5,
                           checkpoints=None,
                           checkpoint_dir='./checkpoints',
                           tracking=False):

        r""" Train posterior using ADVI method.
        (maximising likehood of the data and minimising KL-divergence of posterior to prior)
        :param n: number of independent initialisations
        :param method: to allow for potential use of SVGD or MCMC (currently only ADVI implemented).
        :param n_type: type of repeated initialisation:
                                  'restart' to pick different initial value,
                                  'cv' for molecular cross-validation - splits counts into n datasets,
                                         for now, only n=2 is implemented
                                  'bootstrap' for fitting the model to multiple downsampled datasets.
                                         Run `mod.bootstrap_data()` to generate variants of data
        :param n_iter: number of iterations, supersedes self.n_iter
        :param train_proportion: if not None, which proportion of cells to use for training and which for validation.
        :param checkpoints: int, list of int's or None, number of checkpoints to save while model training or list of
            iterations to save checkpoints on
        :param checkpoint_dir: str, directory to save checkpoints in
        :param tracking: bool, track all latent variables during training - if True makes training 2 times slower
        :return: None
        """

        # initialise parameter store
        self.svi = {}
        self.hist = {}
        self.guide_i = {}
        self.trace_elbo_i = {}
        self.samples = {}
        self.node_samples = {}

        if tracking:
            self.logp_hist = {}

        if n_iter is None:
            n_iter = self.n_iter

        if type(checkpoints) is int:
            if n_iter < checkpoints:
                checkpoints = n_iter
            checkpoints = np.linspace(0, n_iter, checkpoints + 1, dtype=int)[1:]
            self.checkpoints = list(checkpoints)
        else:
            self.checkpoints = checkpoints

        self.checkpoint_dir = checkpoint_dir

        self.n_type = n_type
        self.l2_weight = l2_weight
        self.sample_scaling_weight = sample_scaling_weight
        self.train_proportion = train_proportion

        if stratify_cv is not None:
            self.stratify_cv = stratify_cv

        if train_proportion is not None:
            self.validation_hist = {}
            self.training_hist = {}
            if tracking:
                self.logp_hist_val = {}
                self.logp_hist_train = {}

        if learning_rate is None:
            learning_rate = self.learning_rate

        if np.isin(n_type, ['bootstrap']):
            if self.X_data_sample is None:
                self.bootstrap_data(n=n)
        elif np.isin(n_type, ['cv']):
            self.generate_cv_data()  # cv data added to self.X_data_sample

        init_names = ['init_' + str(i + 1) for i in np.arange(n)]

        for i, name in enumerate(init_names):
            ################### Initialise parameters & optimiser ###################
            # initialise Variational distribution = guide
            if method is 'advi':
                if len(self.point_estim + flatten_iterable(self.custom_guides.keys())) > 0:
                    self.guide_i[name] = AutoGuideList(self.model)
                    normal_guide_block = poutine.block(self.model, expose_all=True, hide_all=False,
                                                       hide=self.point_estim + flatten_iterable(
                                                           self.custom_guides.keys()))
                    self.guide_i[name].append(AutoNormal(normal_guide_block, init_loc_fn=init_to_mean))
                    self.guide_i[name].append(
                        AutoDelta(poutine.block(self.model, hide_all=True, expose=self.point_estim)))
                    for k, v in self.custom_guides.items():
                        self.guide_i[name].append(v)
                else:
                    self.guide_i[name] = AutoNormal(self.model, init_loc_fn=init_to_mean)

                self.guide_type = type(self.guide_i[name]).__name__

            elif method is 'custom':
                self.guide_i[name] = self.guide

            def initialise_svi(x_data, extra_data):

                pyro.clear_param_store()

                self.set_initial_values()

                self.init_guide(name, x_data, extra_data)

                self.trace_elbo_i[name] = JitTrace_ELBO()  # JitTrace_ELBO()

                # initialise SVI inference method
                self.svi[name] = SVI(self.model, self.guide_i[name],
                                     optim.ClippedAdam({'lr': learning_rate,
                                                        # limit the gradient step from becoming too large
                                                        'clip_norm': self.total_grad_norm_constraint}),
                                     loss=self.trace_elbo_i[name])

            # record ELBO Loss history here
            self.hist[name] = []
            if tracking:
                self.logp_hist[name] = defaultdict(list)

            if train_proportion is not None:
                self.validation_hist[name] = []
                if tracking:
                    self.logp_hist_val[name] = defaultdict(list)

            ################### Select data for this iteration ###################
            if np.isin(n_type, ['cv', 'bootstrap']):
                X_data = self.X_data_sample[i].astype(self.data_type)
            else:
                X_data = self.X_data.astype(self.data_type)

            ################### Training / validation split ###################
            # split into training and validation
            if train_proportion is not None:
                idx = np.arange(len(X_data))
                train_idx, val_idx = train_test_split(idx, train_size=train_proportion,
                                                      shuffle=True, stratify=self.stratify_cv)

                extra_data_val = {k: torch.FloatTensor(v[val_idx]).to(self.device) for k, v in self.extra_data.items()}
                extra_data_train = {k: torch.FloatTensor(v[train_idx]) for k, v in self.extra_data.items()}

                x_data_val = torch.FloatTensor(X_data[val_idx]).to(self.device)
                x_data = torch.FloatTensor(X_data[train_idx])
            else:
                # just convert data to CPU tensors
                x_data = torch.FloatTensor(X_data)
                extra_data_train = {k: torch.FloatTensor(v) for k, v in self.extra_data.items()}

            ################### Move data to cuda - FULL data ###################
            # if not minibatch do this:
            if self.minibatch_size is None:
                # move tensors to CUDA
                x_data = x_data.to(self.device)
                for k in extra_data_train.keys():
                    extra_data_train[k] = extra_data_train[k].to(self.device)
                # extra_data_train = {k: v.to(self.device) for k, v in extra_data_train.items()}

            ################### MINIBATCH data ###################
            else:
                # create minibatches
                dataset = MiniBatchDataset(x_data, extra_data_train, return_idx=True)
                loader = DataLoader(dataset, batch_size=self.minibatch_size,
                                    num_workers=0)  # TODO num_workers

            ################### Training the model ###################
            if self.minibatch_size is None:
                initialise_svi(x_data, extra_data_train)
            else:
                i = 0
                for batch in loader:
                    i = i + 1
                    if i == 1:
                        x_data_batch, extra_data_batch = batch
                        x_data_batch = x_data_batch.to(self.device)
                        extra_data_batch = {k: v.to(self.device) for k, v in extra_data_batch.items()}

                initialise_svi(x_data, extra_data_batch)

            # start training in epochs
            epochs_iterator = tqdm(range(n_iter))
            for epoch in epochs_iterator:

                if self.minibatch_size is None:
                    ################### Training FULL data ###################
                    iter_loss = self.step_train(name, x_data, extra_data_train)

                    self.hist[name].append(iter_loss)
                    # save data for posterior sampling
                    self.x_data = x_data
                    self.extra_data_train = extra_data_train

                    if tracking:
                        guide_tr, model_tr = self.step_trace(name, x_data, extra_data_train)
                        self.logp_hist[name]['guide'].append(guide_tr.log_prob_sum().item())
                        self.logp_hist[name]['model'].append(model_tr.log_prob_sum().item())

                        for k, v in model_tr.nodes.items():
                            if "log_prob_sum" in v:
                                self.logp_hist[name][k].append(v["log_prob_sum"].item())

                else:
                    ################### Training MINIBATCH data ###################
                    aver_loss = []
                    if tracking:
                        aver_logp_guide = []
                        aver_logp_model = []
                        aver_logp = defaultdict(list)

                    for batch in loader:

                        x_data_batch, extra_data_batch = batch
                        x_data_batch = x_data_batch.to(self.device)
                        extra_data_batch = {k: v.to(self.device) for k, v in extra_data_batch.items()}

                        loss = self.step_train(name, x_data_batch, extra_data_batch)

                        if tracking:
                            guide_tr, model_tr = self.step_trace(name, x_data_batch, extra_data_batch)
                            aver_logp_guide.append(guide_tr.log_prob_sum().item())
                            aver_logp_model.append(model_tr.log_prob_sum().item())

                            for k, v in model_tr.nodes.items():
                                if "log_prob_sum" in v:
                                    aver_logp[k].append(v["log_prob_sum"].item())

                        aver_loss.append(loss)

                    iter_loss = np.sum(aver_loss)

                    # save data for posterior sampling
                    self.x_data = x_data_batch
                    self.extra_data_train = extra_data_batch

                    self.hist[name].append(iter_loss)

                    if tracking:
                        iter_logp_guide = np.sum(aver_logp_guide)
                        iter_logp_model = np.sum(aver_logp_model)
                        self.logp_hist[name]['guide'].append(iter_logp_guide)
                        self.logp_hist[name]['model'].append(iter_logp_model)

                        for k, v in aver_logp.items():
                            self.logp_hist[name][k].append(np.sum(v))

                if self.checkpoints is not None:
                    if (epoch + 1) in self.checkpoints:
                        self.save_checkpoint(epoch + 1, prefix=name)

                ################### Evaluating cross-validation loss ###################
                if train_proportion is not None:

                    iter_loss_val = self.step_eval_loss(name, x_data_val, extra_data_val)

                    if tracking:
                        guide_tr, model_tr = self.step_trace(name, x_data_val, extra_data_val)
                        self.logp_hist_val[name]['guide'].append(guide_tr.log_prob_sum().item())
                        self.logp_hist_val[name]['model'].append(model_tr.log_prob_sum().item())

                        for k, v in model_tr.nodes.items():
                            if "log_prob_sum" in v:
                                self.logp_hist_val[name][k].append(v["log_prob_sum"].item())

                    self.validation_hist[name].append(iter_loss_val)
                    epochs_iterator.set_description(f'ELBO Loss: ' + '{:.4e}'.format(iter_loss) \
                                                    + ': Val loss: ' + '{:.4e}'.format(iter_loss_val))
                else:
                    epochs_iterator.set_description('ELBO Loss: ' + '{:.4e}'.format(iter_loss))

                if epoch % 20 == 0:
                    torch.cuda.empty_cache()

            if train_proportion is not None:
                # rescale loss
                self.validation_hist[name] = [i / (1 - train_proportion)
                                              for i in self.validation_hist[name]]
                self.hist[name] = [i / train_proportion for i in self.hist[name]]

                # reassing the main loss to be displayed
                self.training_hist[name] = self.hist[name]
                self.hist[name] = self.validation_hist[name]

                if tracking:
                    for k, v in self.logp_hist[name].items():
                        self.logp_hist[name][k] = [i / train_proportion for i in self.logp_hist[name][k]]
                        self.logp_hist_val[name][k] = [i / (1 - train_proportion) for i in self.logp_hist_val[name][k]]

                    self.logp_hist_train[name] = self.logp_hist[name]
                    self.logp_hist[name] = self.logp_hist_val[name]

            if self.verbose:
                print(plt.plot(np.log10(self.hist[name][0:])));

    def init_guide(self, name, x_data, extra_data):

        self.guide_i[name](x_data)

    def step_train(self, name, x_data, extra_data):

        return self.svi[name].step(x_data)

    def step_eval_loss(self, name, x_data, extra_data):

        return self.svi[name].evaluate_loss(x_data)

    def step_predictive(self, predictive, x_data, extra_data):

        return predictive(x_data)

    def step_trace(self, name, x_data, extra_data):

        guide_tr = poutine.trace(self.guide_i[name]).get_trace(x_data)
        model_tr = poutine.trace(poutine.replay(self.model,
                                                trace=guide_tr)).get_trace(x_data)
        return guide_tr, model_tr

    def fit_nuts(self, n_samples: int = 1000, warmup_steps: int = 1000, save_samples=False):

        self.samples = {}
        self.n_samples = n_samples

        # create sampler and run MCMC
        self.nuts_kernel = NUTS(self.model, jit_compile=True)
        self.mcmc = MCMC(self.nuts_kernel, num_samples=n_samples,
                         warmup_steps=warmup_steps)
        self.mcmc.run(self.x_data)

        post_samples = {k: v.detach().cpu().numpy() for k, v in self.mcmc.get_samples().items()}

        # summarise samples
        self.samples['post_sample_means'] = {v: post_samples[v].mean(axis=0) for v in post_samples.varnames}
        self.samples['post_sample_q05'] = {v: np.quantile(post_samples[v], 0.05, axis=0) for v in post_samples.varnames}
        self.samples['post_sample_q95'] = {v: np.quantile(post_samples[v], 0.95, axis=0) for v in post_samples.varnames}
        self.samples['post_sample_sds'] = {v: post_samples[v].std(axis=0) for v in post_samples.varnames}

        if (save_samples):
            self.samples['post_samples'] = post_samples

    def plot_history_old(self, iter_start: int = 15000, iter_end=-1):
        r""" Plot training history
        :param iter_start: omit initial iterations from the plot
        :param iter_end: omit last iterations from the plot
        """
        for i in self.hist.keys():
            print(plt.plot(np.log10(np.array(self.hist[i])[iter_start:iter_end])));

    def plot_history_1(self, iter_start=0, iter_end=-1,
                       mean_field_slot=None, log_y=True, ax=None):
        r""" Plot training history

        :param iter_start: omit initial iterations from the plot
        :param iter_end: omit last iterations from the plot
        """

        if ax is None:
            ax = plt
            ax.set_xlabel = plt.xlabel
            ax.set_ylabel = plt.ylabel

        if mean_field_slot is None:
            mean_field_slot = self.hist.keys()

        for i in mean_field_slot:

            if iter_end == -1:
                iter_end = np.array(self.hist[i]).flatten().shape[0]

            y = np.array(self.hist[i]).flatten()[iter_start:iter_end]
            if log_y:
                y = np.log10(y)
            ax.plot(np.arange(iter_start, iter_end), y, label='train')
            ax.set_xlabel('Training epochs')
            ax.set_ylabel('Reconstruction accuracy (ELBO loss)')
            ax.legend()
            plt.tight_layout()

    def sample_node1(self, node, init, batch_size: int = 10):

        predictive = Predictive(self.model, guide=self.guide_i[init],
                                num_samples=batch_size)

        post_samples = {k: v.detach().cpu().numpy()
                        for k, v in self.step_predictive(predictive, self.x_data, self.extra_data_train).items()
                        if k == node}

        return (post_samples[node])

    def sample_node(self, node, init, n_sampl_iter,
                    batch_size: int = 10, suff=''):

        # sample first batch
        self.samples[node + suff][init] = self.sample_node1(node, init, batch_size=batch_size)

        for it in tqdm(range(n_sampl_iter - 1)):
            # sample remaining batches
            post_node = self.sample_node1(node, init, batch_size=batch_size)

            # concatenate batches
            self.samples[node + suff][init] = np.concatenate((self.samples[node + suff][init], post_node), axis=0)

        # compute mean across samples
        self.samples[node + suff][init] = self.samples[node + suff][init].mean(0)

    def sample_all1(self, init='init_1', batch_size: int = 10):

        predictive = Predictive(self.model, guide=self.guide_i[init],
                                num_samples=batch_size)

        post_samples = {k: v.detach().cpu().numpy()
                        for k, v in self.step_predictive(predictive, self.x_data, self.extra_data_train).items()
                        if k != "data_target"}

        return (post_samples)

    def sample_all(self, n_sampl_iter, init='init_1', batch_size: int = 10):

        # sample first batch
        self.samples['post_samples'] = self.sample_all1(init, batch_size=batch_size)

        for it in tqdm(range(n_sampl_iter - 1)):
            # sample remaining batches
            post_samples = self.sample_all1(init, batch_size=batch_size)

            # concatenate batches
            self.samples['post_samples'] = {k: np.concatenate((self.samples['post_samples'][k],
                                                               post_samples[k]), axis=0)
                                            for k in post_samples.keys()}

    def b_evaluate_stability(self, node, n_samples: int = 1000, batch_size: int = 10,
                             align=True, transpose=True):
        r""" Evaluate stability of posterior samples between training initialisations
        (takes samples and correlates the values of factors between training initialisations)
        :param node: which pymc3 node to sample? Factors should be in columns.
        :param n_samples: the number of samples.
        :param batch_size: generate samples in batches of size `batch_size`. Necessary for the computation to fit in the GPU memory 
        :return: self.samples[node_name+_stab] dictionary with an element for each training initialisation. 
        """

        self.n_samples = n_samples
        self.n_sampl_iter = int(np.ceil(n_samples / batch_size))
        self.n_sampl_batch = batch_size

        self.samples[node + '_stab'] = {}

        for i in self.guide_i.keys():
            self.sample_node(node, i, self.n_sampl_iter,
                             batch_size=self.n_sampl_batch, suff='_stab')

        # plot correlations of posterior mean between training initialisations
        for i in range(len(self.samples[node + '_stab'].keys()) - 1):
            x = self.samples[node + '_stab']['init_' + str(1)]
            y = self.samples[node + '_stab']['init_' + str(i + 2)]
            if transpose:
                x = x.T
                y = y.T
            print(self.align_plot_stability(x, y,
                                            str(1), str(i + 2), align=align))

    def sample_posterior(self, node='all',
                         n_samples: int = 1000, batch_size: int = 10,
                         save_samples=False,
                         mean_field_slot='init_1'):
        r""" Sample posterior distribution of parameters - either all or single parameter
        :param node: pyro parameter to sample (e.g. default "all", self.spot_factors)
        :param n_samples: number of posterior samples to generate (1000 is recommended, reduce if you get GPU memory error)
        :param save_samples: save samples in addition to sample mean, 5% quantile, SD.
        :param return_samples: return summarised samples in addition to saving them in `self.samples`
        :param mean_field_slot: string, which mean_field slot to sample? 'init_1' by default
        :return: dictionary of dictionaries (mean, 5% quantile, SD, optionally all samples) with numpy arrays for each parameter.
        Optional dictionary of all samples contains parameters as numpy arrays of shape ``(n_samples, ...)``
        """

        self.n_samples = n_samples
        self.n_sampl_iter = int(np.ceil(n_samples / batch_size))
        self.n_sampl_batch = batch_size

        if (node == 'all'):
            # Sample all parameters - might use a lot of GPU memory

            self.sample_all(self.n_sampl_iter, init=mean_field_slot, batch_size=self.n_sampl_batch)

            self.param_names = list(self.samples['post_samples'].keys())

            self.samples['post_sample_means'] = {v: self.samples['post_samples'][v].mean(axis=0)
                                                 for v in self.param_names}
            self.samples['post_sample_q05'] = {v: np.quantile(self.samples['post_samples'][v], 0.05, axis=0)
                                               for v in self.param_names}
            self.samples['post_sample_q95'] = {v: np.quantile(self.samples['post_samples'][v], 0.95, axis=0)
                                               for v in self.param_names}
            self.samples['post_sample_sds'] = {v: self.samples['post_samples'][v].std(axis=0)
                                               for v in self.param_names}

            if not save_samples:
                self.samples['post_samples'] = None

        else:
            self.sample_node(node, mean_field_slot, self.n_sampl_iter,
                             batch_size=self.n_sampl_batch, suff='')

        return (self.samples)

    def save_checkpoint(self, n, prefix=''):
        r""" Save pyro parameter store (current status of Variational parameters) to disk
        :param n: epoch number
        :param prefix: filename prefix (e.g. init number)
        """

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        filename = f'{self.checkpoint_dir}/{prefix}_{n}.ckp'
        pyro.get_param_store().save(filename)

    def load_checkpoint(self, filename):
        r""" Load pyro parameter store (current status of Variational parameters) from disk
        :param filename: checkpoint filename
        """

        if filename in os.listdir(self.checkpoint_dir):
            pyro.get_param_store().load(filename)
        else:
            checkpoints = os.listdir(self.checkpoint_dir)
            checkpoints = '\n'.join(checkpoints)
            checkpoint_dir_abspath = os.path.abspath(self.checkpoint_dir)
            raise ValueError(f'No such filename in {checkpoint_dir_abspath}, available filenames : \n'
                             f'{checkpoints}')
