# +
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pycell2location.models.base_model import BaseModel
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader


class MiniBatchDataset(Dataset):

    def __init__(self, x_data, extra_data):
        self.x_data = x_data
        self.extra_data = extra_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return (self.x_data[idx], *(data[idx] for data in self.extra_data.values()))



class TorchModel(BaseModel):

    def __init__(
            self,
            X_data: np.ndarray,
            n_fact: int = 10,
            data_type: str = 'float32',
            n_iter: int = 200000,
            learning_rate=0.001,
            total_grad_norm_constraint=200,
            verbose=True,
            var_names=None, var_names_read=None,
            obs_names=None, fact_names=None, sample_id=None,
            use_cuda=False
    ):

        ############# Initialise parameters ################
        super().__init__(X_data, n_fact,
                         data_type, n_iter,
                         learning_rate, total_grad_norm_constraint,
                         verbose, var_names, var_names_read,
                         obs_names, fact_names, sample_id)

        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.extra_data = {} # if no extra data empty dictionary
        
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

    # =====================DATA likelihood loss functions======================= #
    # define cost function 
    def nb_log_prob(self, param, data, eps=1e-8):
        """ Method that returns log probability / log likelihood for each data point.
        NB log prob - Copied over from scVI 
        https://github.com/YosefLab/scVI/blob/b20e34f02a87d16790dbacc95b2ae1714c08615c/scvi/models/log_likelihood.py#L249
        Note: All inputs should be torch Tensors
        log likelihood (scalar) of a minibatch according to a nb model.
        Variables:
        :param param: list with [mu, theta]
        :param data: data (cells * genes)
        mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
        theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
        eps: numerical stability constant
        """
        mu = param[0]
        theta = param[1]

        log_theta_mu_eps = torch.log(theta + mu + eps)

        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + data * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(data + theta)
            - torch.lgamma(theta)
            - torch.lgamma(data + 1)
        )

        return res

    # =====================Training functions======================= #
    def fit_advi_iterative(self, n=3, method='advi', n_type='restart',
                           n_iter=None, learning_rate=None,
                           progressbar=True, num_workers=2, train_proportion=None,
                           l2_weight=False, sample_scaling_weight=0.5):
        r""" Find posterior using pm.ADVI() method directly (allows continuing training through `refine` method.
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
        :return: self.mean_field dictionary with MeanField pymc3 objects, and self.advi dictionary with ADVI objects.
        """

        self.mean_field = {}
        self.hist = {}
        self.validation_hist = {}
        self.training_hist = {}
        self.samples = {}
        self.node_samples = {}


        self.n_type = n_type
        self.l2_weight = l2_weight
        self.sample_scaling_weight = sample_scaling_weight
        self.train_proportion = train_proportion

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
            ################### Initialise parameters & optimiser ###################
            self.model.initialize_parameters()
            optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.hist[name] = []
            self.validation_hist[name] = []

            ################### Move data to pytorch - FULL data ###################
            # convert data to CPU tensors
            if np.isin(n_type, ['cv', 'bootstrap']):
                x_data = torch.FloatTensor(self.X_data_sample[i].astype(self.data_type))
            else:
                x_data = torch.FloatTensor(self.X_data.astype(self.data_type))


            ################### Training / validation split ###################
            # split into training and validation
            if train_proportion is not None:
                idx = np.arange(len(x_data))
                train_idx, val_idx = train_test_split(idx, train_size=train_proportion,
                                                      shuffle=True, stratify=self.stratify_cv)

                extra_data_val = {k: torch.FloatTensor(v[val_idx]).to(self.device) for k, v in self.extra_data.items()}
                extra_data_train = {k: torch.FloatTensor(v[train_idx]) for k, v in self.extra_data.items()}

                x_data_val = x_data[val_idx].to(self.device)
                x_data = x_data[train_idx]


            ################### Move data to cuda - FULL data ###################
            # if not minibatch do this:
            if self.minibatch_size is None:
                x_data.to(self.device)
                extra_data_train = {k: v.to(self.device) for k, v in extra_data_train.items()}

            ################### MINIBATCH data ###################
            else:
                # create minibatch
                dataset = MiniBatchDataset(x_data, extra_data_train)
                loader = DataLoader(dataset, batch_size=self.minibatch_size,
                                    num_workers=num_workers)

            ################### Training the model ###################
            # start training in epochs
            epochs_iterator = tqdm(range(n_iter))
            for epoch in epochs_iterator:

                if self.minibatch_size is None:
                    ################### Training FULL data ###################
                    self.model.train()
                    optim.zero_grad()
                    y_pred = self.model.forward(**extra_data_train)
                    loss = self.loss(y_pred, x_data, l2_weight=l2_weight,
                                     sample_scaling_weight=sample_scaling_weight)
                    loss.backward()
                    optim.step()
                    iter_loss=loss.item()

                else:
                    ################### Training MINIBATCH data ###################
                    aver_loss = []

                    for batch in loader:
                        self.model.train()
                        x_data_batch, *extra_data_batch = batch

                        x_data_batch = x_data_batch.to(self.device)
                        extra_data_batch = {k:v.to(self.device) for k, v in zip(extra_data_train.keys(), extra_data_batch)}

                        optim.zero_grad()
                        y_pred = self.model.forward(**extra_data_batch)
                        loss = self.loss(y_pred, x_data_batch, l2_weight=l2_weight,
                                     sample_scaling_weight=sample_scaling_weight)
                        aver_loss.append(loss.item())
                        loss.backward()

                        optim.step()

                    iter_loss = np.sum(aver_loss)

                self.hist[name].append(iter_loss)

                ################### Evaluating cross-validation loss ###################
                if train_proportion is not None:

                    aver_loss_val = []

                    self.model.eval()
                    y_pred = self.model.forward(**extra_data_val)
                    loss = self.loss(y_pred, x_data_val, l2_weight=l2_weight,
                                     sample_scaling_weight=sample_scaling_weight)
                    aver_loss_val.append(loss.item())
                    iter_loss_val = np.sum(aver_loss_val)

                    self.validation_hist[name].append(iter_loss_val)
                    epochs_iterator.set_description(f'Loss: '+'{:.4e}'.format(iter_loss)
                                                    +': Val loss: '+ '{:.4e}'.format(iter_loss_val))
                else:
                    epochs_iterator.set_description('Loss: ' + '{:.4e}'.format(iter_loss))


            if train_proportion is not None:
                # rescale loss
                self.validation_hist[name] = [i / (1 - train_proportion)
                                              for i in self.validation_hist[name]]
                self.hist[name] = [i / train_proportion for i in self.hist[name]]
                # reassing the main loss to be displayed
                self.training_hist[name] = self.hist[name]
                self.hist[name] = self.validation_hist[name]

            ################## Exporting parameters ###################
            self.mean_field[name] = self.model.state_dict()

            if self.verbose:
                print(plt.plot(np.log10(self.hist[name][0:])));

    def sample_posterior(self, node='all', n_samples=1000,
                         save_samples=False, return_samples=True,
                         mean_field_slot='init_1'):
        r""" Sample posterior distribution of parameters - either all or single parameter
        :param node: torch parameter to sample (e.g. default "all", self.spot_factors)
        :param n_samples: number of posterior samples to generate (1000 is recommended, reduce if you get GPU memory error)
        :param save_samples: save samples in addition to sample mean, 5% quantile, SD.
        :param return_samples: return summarised samples in addition to saving them in `self.samples`
        :param mean_field_slot: string, which mean_field slot to sample? 'init_1' by default
        :return: dictionary of dictionaries (mean, 5% quantile, SD, optionally all samples) with numpy arrays for each parameter.
        Optional dictionary of all samples contains parameters as numpy arrays of shape ``(n_samples, ...)``
        """

        self.model.load_state_dict(self.mean_field[mean_field_slot])

        if node == 'all':
            post_samples = self.model.export_parameters()
            post_samples = {v: post_samples[v].cpu().detach().numpy()
                            for v in post_samples.keys()}
            self.samples['post_sample_means'] = post_samples

            if save_samples:
                self.samples['post_samples'] = post_samples

        else:
            raise NotImplementedError
            
        self.samples['post_sample_sds'] = None
        self.samples['post_sample_q05'] = None
        self.samples['post_sample_q95'] = None

        if return_samples:
            return self.samples

    def b_evaluate_stability(self, node_name, n_samples=1000, align=True, transpose=True):
        r""" Evaluate stability of posterior samples between training initialisations
        (takes samples and correlates the values of factors between training initialisations)
        :param node: which pymc3 node to sample? Factors should be in columns.
        :param n_samples: the number of samples.
        :param align: boolean, match factors between training restarts using linear_sum_assignment?
        :return: self.samples[node_name+_stab] dictionary with an element for each training initialisation.
        """

        self.n_samples = n_samples

        self.samples[node_name + '_stab'] = {}

        for i in self.mean_field.keys():
            self.model.load_state_dict(self.mean_field[i])
            post_samples = self.model.export_parameters()[node_name].cpu().detach().numpy()
            if transpose:
                post_samples = post_samples.T
            self.samples[node_name + '_stab'][i] = post_samples

        for i in range(len(self.samples[node_name + '_stab'].keys()) - 1):
            print(self.align_plot_stability(self.samples[node_name + '_stab']['init_' + str(1)],
                                            self.samples[node_name + '_stab']['init_' + str(i + 2)],
                                            str(1), str(i + 2), align=align))

    def sample2df(self, node_name='nUMI_factors'):
        r""" Export spot factors as Pandas data frames.
        :param node_name: name of the model parameter to be exported
        :return: a Pandas dataframe added to model object:
                 .spot_factors_df
        """

        if len(self.samples) == 0:
            raise ValueError(
                'Please run `.sample_posterior()` first to generate samples & summarise posterior of each parameter')

        self.spot_factors_df = \
            pd.DataFrame.from_records(self.samples['post_sample_means'][node_name],
                                      index=self.obs_names,
                                      columns=['mean_' + node_name + i for i in self.fact_names])

    def annotate_spot_adata(self, adata):
        r""" Add spot factors to anndata.obs
        :param adata: anndata object to annotate
        :return: updated anndata object
        """

        if self.spot_factors_df is None:
            self.sample2df()

        # add cell factors to adata
        adata.obs[self.spot_factors_df.columns] = self.spot_factors_df.loc[adata.obs.index, :]

        return (adata)
