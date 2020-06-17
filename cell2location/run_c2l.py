# -*- coding: utf-8 -*-
"""Run full pipeline for locating cells with cell2location model."""

import gc
import os
import pickle
# +
import time
from os import mkdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano

import cell2location.plt as c2lpl
from cell2location.cluster_averages import get_cluster_averages
from cell2location.cluster_averages import select_features


def save_plot(path, filename, extension='png'):
    r""" Save current plot to `path` as `filename` with `extension`
    """

    plt.savefig(path + filename + '.' + extension)
    # fig.clear()
    plt.close()


def run_cell2location(sc_data, sp_data, model_name='CoLocationModelNB4V2',
                      verbose=True, return_all=True,
                      summ_sc_data_args={'cluster_col': "annotation_1"},
                      train_args={'n_iter': 20000, 'learning_rate': 0.005,
                                  'sample_prior': False, 'readable_var_name_col': None,
                                  'sample_name_col': None},
                      model_kwargs={},
                      posterior_args={'n_samples': 1000},
                      export_args={'path': "./results", 'save_model': False,
                                   'run_name_suffix': '', 'scanpy_coords_name': 'coords'}):
    r""" Run cell2location model pipeline: train, sample prior and posterior, export results and save diagnostic plots
    Spatial expression of cell types (proxy for density) is exported as columns of adata.obs,
       named `'mean_nUMI_factors' + 'cluster name'` for posterior mean or 
             `'q05_nUMI_factors' + 'cluster name'` for 5% quantile of the posterior
    and as np.ndarray in `adata.uns['mod']['post_sample_means']['nUMI_factors']` (also post_sample_q05, post_sample_q95, post_sample_sd)
    Anndata object with exported results, W weights representing cell state densities, and the trained model object are saved to `export_args['path']`
    
    :param sc_data: anndata object with single cell / nucleus data, 
                        or pd.DataFrame with genes in rows and cell type signatures (factors) in columns.
    :param sp_data: anndata object with spatial data, variable names should match sc_data
    :param model: model name as a string
    :param summ_sc_data_args: arguments for summarising single cell data
                'cluster_col' - name of sc_data.obs column containing cluster annotations
                'which_genes' - select intersect or union genes between single cell and spatial?
                'selection' - select highly variable genes in sc_data? By default no (None), availlable options:
                                None
                                "high_cv" use high coefficient of variation (var / mean)
                                "cluster_markers" use cluster markers (derived using sc.tl.rank_genes_groups)
                                "AutoGeneS" use https://github.com/theislab/AutoGeneS
                'select_n' - how many variablegenes to select?
                'select_n_AutoGeneS' - how many variable genes tor select with AutoGeneS (lower than select_n)?
    :param train_args: arguments for training methods. See help(c2l.LocationModel) for more details
                'mode' - "normal" or "tracking" parameters?
                'use_raw' - extract data from RAW slot of anndata? Applies only to spatial, not single cell reference.
                'sample_prior' - use sampling from the prior to evaluate how reasonable priors are for your data. 
                                 This is not essential for Visium data but can be very useful for troubleshooting. 
                                 It is essential for choose priors appropriate for any other spatial technology.
                                 Caution: takes a lot of RAM (10th-100s of GB).
                'n_prior_samples' - Number of prior sample. The more the better but also takes a lot of RAM.
                'n_restarts' - number of training restarts to evaluate stability
                'n_type' - type of restart training: 'restart' 'cv' 'bootstrap'(see help(c2l.LocationModel.fit_advi_iterative) for details)
                'method' - which method to use to find posterior. Use default 'advi' unless you know what you are doing.
                'readable_var_name_col' - column in sp_data.var that contains readable gene ID (e.g. HGNC symbol)
                'sample_name_col' - column in sp_data.obs that contains sample / slide ID (e.g. Visium section) 
                                    - for plotting W cell locations
                'fact_names' - optional list of factor names, by default taken from sc_data.
    :param posterior_args: arguments for sampling posterior, 
                We get 1000 samples from the posterior distribution of each parameter 
                to compute means, SD, 5% and 95% quantiles - stored in `mod.samples` and exported in `adata.uns['mod']`.
                'n_samples' - number of samples to get from posterior approximation to compute average and SD of that distribution
    :param export_args: arguments for exporting results
                'path' - path where to save results
                'plot_extension' - file extention of saved plots
                'save_model' - boolean, save trained model? Could be useful but also takes up GBs of disk.
                'export_q05' - boolean, save plots of 5% quantile of parameters.
    :param return_all: return the model and annotated `sp_data` or just save both?
    :return: results as a dictionary {'mod','sp_data','sc_obs','model_name',
                                      'summ_sc_data_args', 'train_args','posterior_args', 'export_args',
                                      'run_name', 'run_time'}
    """

    # set default parameters
    d_summ_sc_data_args = {'cluster_col': "annotation_1", 'which_genes': "intersect",
                           'selection': None, 'select_n': 5000, 'select_n_AutoGeneS': 1000}

    d_train_args = {'mode': "normal", 'use_raw': True, 'data_type': "float32",
                    'n_iter': 20000, 'learning_rate': 0.005, 'total_grad_norm_constraint': 200,
                    'method': 'advi',
                    'sample_prior': False, 'n_prior_samples': 10,
                    'n_restarts': 2, 'n_type': "restart",
                    'tracking_every': 1000, 'tracking_n_samples': 50, 'readable_var_name_col': None,
                    'sample_name_col': None, 'fact_names': None}

    d_posterior_args = {'n_samples': 1000, 'evaluate_stability_align': False, 'mean_field_slot': "init_1"}

    d_export_args = {'path': "./results", 'plot_extension': "png",
                     'save_model': False, 'run_name_suffix': '', 'export_q05': True,
                     'scanpy_coords_name': 'spatial'}

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

    sp_data = sp_data.copy()

    # move spatial coordinates to obs for compatibility with our plotter
    sp_data.obs['imagecol'] = sp_data.obsm[export_args['scanpy_coords_name']][:, 0]
    sp_data.obs['imagerow'] = sp_data.obsm[export_args['scanpy_coords_name']][:, 1]

    # import the specied version of the model
    if type(model_name) is str:
        import cell2location.models as models
        Model = getattr(models, model_name)
    else:
        Model = model_name

    ####### Summarising single cell clusters #######
    if verbose:
        print('### Summarising single cell clusters ###')

    if not isinstance(sc_data, pd.DataFrame):
        # if scanpy compute cluster averages
        cell_state_df = get_cluster_averages(sc_data, cluster_col=summ_sc_data_args['cluster_col'])
        obs = sc_data.obs
    else:
        # if dataframe assume signature with matching .index to sp_data.var_names
        cell_state_df = sc_data.copy()
        obs = cell_state_df

    # select features if requested
    if summ_sc_data_args['selection'] == 'high_cv':

        cv = cell_state_df.var(1) / cell_state_df.mean(1)
        cv = cv.sort_values(ascending=False)
        cell_state_df = cell_state_df.loc[cv[np.arange(summ_sc_data_args['select_n'])].index, :]

    elif summ_sc_data_args['selection'] == 'cluster_markers':
        if isinstance(sc_data, pd.DataFrame):
            raise ValueError(
                'summ_sc_data_args["selection"] = "cluster_markers" can only be used with type(sc_data) Anndata')

        sel_feat = select_features(sc_data, summ_sc_data_args['cluster_col'],
                                   n_features=summ_sc_data_args['select_n'], use_raw=train_args['use_raw'])
        cell_state_df = cell_state_df.loc[cell_state_df.index.isin(sel_feat), :]

    elif summ_sc_data_args['selection'] == 'AutoGeneS':

        cv = cell_state_df.var(1) / cell_state_df.mean(1)
        cv = cv.sort_values(ascending=False)
        cell_state_df = cell_state_df.loc[cv[np.arange(summ_sc_data_args['select_n'])].index, :]

        from autogenes import AutoGenes
        ag = AutoGenes(cell_state_df.T)
        ag.run(ngen=5000, seed=0, nfeatures=summ_sc_data_args['select_n_AutoGeneS'], mode='fixed')
        pareto = ag.pareto
        print(ag.plot(size='large', weights=(1, -1)))
        # We then pick one solution and filter its corresponding marker genes:
        cell_state_df = cell_state_df[pareto[len(pareto) - 1]]  # select the solution with min correlation

    elif summ_sc_data_args['selection'] is not None:
        raise ValueError("summ_sc_data_args['selection'] can be only None, high_cv, cluster_markers or AutoGeneS")

    # extract data as a dense matrix
    if train_args['use_raw']:
        X_data = sp_data.raw.X.toarray()
    else:
        X_data = sp_data.X.toarray()

    # Filter cell states and X_data to common genes
    sp_ind = sp_data.var_names.isin(cell_state_df.index)
    if np.sum(sp_ind) == 0:
        raise ValueError('No overlapping genes found, check that `sc_data` and `sp_data` use the same variable names')
    X_data = X_data[:, sp_ind]
    cell_state_df = cell_state_df.loc[sp_data.var_names[sp_ind], :]

    # prepare cell state matrix
    cell_state_mat = cell_state_df.values

    # if factor names not provided use cluster names
    if train_args['fact_names'] is None:
        fact_names = cell_state_df.columns
    else:  # useful for unobserved factor models
        fact_names = train_args['fact_names']

    if train_args['sample_name_col'] is None:
        sp_data.obs['sample'] = 'sample'
        train_args['sample_name_col'] = 'sample'

    if train_args['readable_var_name_col'] is not None:
        readable_var_name_col = sp_data.var[train_args['readable_var_name_col']][sp_ind]
    else:
        readable_var_name_col = None

    ####### Creating model #######
    if verbose:
        print('### Creating model ### - time ' + str(np.around((time.time() - start) / 60, 2)) + ' min')
    mod = Model(cell_state_mat, X_data,
                data_type=train_args['data_type'], n_iter=train_args['n_iter'],
                learning_rate=train_args['learning_rate'],
                total_grad_norm_constraint=train_args['total_grad_norm_constraint'],
                verbose=verbose,
                var_names=sp_data.var_names[sp_ind],
                var_names_read=readable_var_name_col,
                obs_names=sp_data.obs_names,
                fact_names=fact_names,
                sample_id=sp_data.obs[train_args['sample_name_col']],
                **model_kwargs)

    ####### Print run name #######
    run_name = str(mod.__class__.__name__) + '_' + str(mod.n_fact) + 'clusters_' \
               + str(mod.n_cells) + 'locations_' + str(mod.n_genes) + 'genes' \
               + export_args['run_name_suffix']

    print('### Analysis name: ' + run_name)  # analysis name is always printed

    # create the export directory
    path = export_args['path'] + run_name + '/'
    if not os.path.exists(path):
        mkdir(path)

    fig_path = path + 'plots/'
    if not os.path.exists(fig_path):
        mkdir(fig_path)

    ####### Sampling prior #######
    if train_args['sample_prior']:
        if verbose:
            print('### Sampling prior ###')

        mod.sample_prior(samples=train_args['n_prior_samples'])

        # plot & save plot
        mod.plot_prior_vs_data()
        save_plot(fig_path, filename='evaluating_prior', extension=export_args['plot_extension'])
        plt.close()

    ####### Training model #######
    if verbose:
        print('### Training model ###')
    if train_args['mode'] == 'normal':
        mod.fit_advi_iterative(n=train_args['n_restarts'], method=train_args['method'],
                               n_type=train_args['n_type'], progressbar=verbose)

    elif train_args['mode'] == 'tracking':
        mod.verbose = False
        mod.track_parameters(n=train_args['n_restarts'],
                             every=train_args['tracking_every'], n_samples=train_args['tracking_n_samples'],
                             n_type=train_args['n_type'],
                             df_node_name1='nUMI_factors', df_node_df_name1='spot_factors_df',
                             df_prior_node_name1='spot_fact_mu_hyp', df_prior_node_name2='spot_fact_sd_hyp',
                             mu_node_name='mu', data_node='X_data',
                             extra_df_parameters=['spot_add'],
                             sample_type='post_sample_means')

    else:
        raise ValueError("train_args['mode'] can be only 'normal' or 'tracking'")

    theano.config.compute_test_value = 'ignore'
    ####### Evaluate stability of training #######
    if train_args['n_restarts'] > 1:
        mod.evaluate_stability(n_samples=posterior_args['n_samples'],
                               align=posterior_args['evaluate_stability_align'])
        save_plot(fig_path, filename='evaluate_stability', extension=export_args['plot_extension'])
        plt.close()

    ####### Sampling posterior #######
    if verbose:
        print('### Sampling posterior ### - time ' + str(np.around((time.time() - start) / 60, 2)) + ' min')
    mod.sample_posterior(node='all', n_samples=posterior_args['n_samples'],
                         save_samples=False, mean_field_slot=posterior_args['mean_field_slot']);

    # evaluate predictive accuracy
    mod.compute_expected()

    ####### Export summarised posterior & Saving results #######
    if verbose:
        print('### Saving results ###')

    # save W cell locations (cell density)
    # convert cell location parameters (cell density) to a dataframe
    mod.sample2df(node_name='spot_factors')
    # add cell location parameters to `sp_data.obs`
    sp_data = mod.annotate_spot_adata(sp_data)
    # save
    mod.spot_factors_df.to_csv(path + 'W_cell_density.csv')
    if export_args['export_q05']:
        mod.spot_factors_q05.to_csv(path + 'W_cell_density_q05.csv')

    # convert cell location parameters (mRNA count) to a dataframe, see help(mod.sample2df) for details
    mod.sample2df(node_name='nUMI_factors')

    # add cell location parameters to `sp_data.obs`
    sp_data = mod.annotate_spot_adata(sp_data)

    # save W cell locations (mRNA count)
    mod.spot_factors_df.to_csv(path + 'W_mRNA_count.csv')
    if export_args['export_q05']:
        mod.spot_factors_q05.to_csv(path + 'W_mRNA_count_q05.csv')

    # add posterior of all parameters to `sp_data.uns['mod']` 
    mod.fact_filt = None
    sp_data = mod.export2adata(adata=sp_data, slot_name='mod')
    sp_data.uns['mod']['fact_names'] = list(sp_data.uns['mod']['fact_names'])
    sp_data.uns['mod']['var_names'] = list(sp_data.uns['mod']['var_names'])
    sp_data.uns['mod']['obs_names'] = list(sp_data.uns['mod']['obs_names'])

    # save spatial anndata with exported posterior
    sp_data.write(filename=path + 'sp.h5ad', compression='gzip')

    # save model object and related annotations    
    if export_args['save_model']:
        # save the model and other objects
        res_dict = {'mod': mod, 'sp_data': sp_data, 'sc_obs': obs,
                    'model_name': model_name, 'summ_sc_data_args': summ_sc_data_args,
                    'train_args': train_args, 'posterior_args': posterior_args,
                    'export_args': export_args, 'run_name': run_name,
                    'run_time': str(np.around((time.time() - start) / 60, 2)) + ' min'}
        pickle.dump(res_dict, file=open(path + 'model_.p', "wb"))

    else:
        # just save the settings
        res_dict = {'sc_obs': obs, 'model_name': model_name,
                    'summ_sc_data_args': summ_sc_data_args,
                    'train_args': train_args, 'posterior_args': posterior_args,
                    'export_args': export_args, 'run_name': run_name,
                    'run_time': str(np.around((time.time() - start) / 60, 2)) + ' min'}
        pickle.dump(res_dict, file=open(path + 'model_.p', "wb"))

    ####### Plotting #######
    if verbose:
        print('### Ploting results ###')

    # Show training history #
    if verbose:
        print(mod.plot_history(0))
    else:
        mod.plot_history(0)
    save_plot(fig_path, filename='training_history_all', extension=export_args['plot_extension'])
    plt.close()

    if verbose:
        print(mod.plot_history(int(np.ceil(train_args['n_iter'] * 0.2))))
    else:
        mod.plot_history(int(np.ceil(train_args['n_iter'] * 0.2)))
    save_plot(fig_path, filename='training_history_without_first_20perc',
              extension=export_args['plot_extension'])
    plt.close()

    # Predictive accuracy 
    try:
        mod.plot_posterior_mu_vs_data()
        save_plot(fig_path, filename='data_vs_posterior_mean_Poisson_rate',
                  extension=export_args['plot_extension'])
        plt.close()
    except Exception as e:
        print('Some error in plotting `mod.plot_posterior_mu_vs_data()`\n ' + str(e))

    ####### Ploting posterior of W / cell locations #######
    if verbose:
        print('### Ploting posterior of W / cell locations ###')

    data_samples = sp_data.obs[train_args['sample_name_col']].unique()
    cluster_plot_names = mod.spot_factors_df.columns
    cluster_plot_names = pd.Series([i[17:] for i in mod.spot_factors_df.columns])

    try:
        for i in data_samples:
            p = c2lpl.plot_factor_spatial(adata=sp_data,
                                          fact_ind=np.arange(mod.spot_factors_df.shape[1]),
                                          fact=mod.spot_factors_df,
                                          cluster_names=cluster_plot_names,
                                          n_columns=6, trans='log',
                                          sample_name=i, samples_col=train_args['sample_name_col'],
                                          obs_x='imagecol', obs_y='imagerow')
            p.save(filename=fig_path + 'cell_locations_W_mRNA_count_' + str(i) + '.' + export_args['plot_extension'])

            if export_args['export_q05']:
                p = c2lpl.plot_factor_spatial(adata=sp_data,
                                              fact_ind=np.arange(mod.spot_factors_q05.shape[1]),
                                              fact=mod.spot_factors_q05,
                                              cluster_names=cluster_plot_names,
                                              n_columns=6, trans='log',
                                              sample_name=i, samples_col=train_args['sample_name_col'],
                                              obs_x='imagecol', obs_y='imagerow')
                p.save(filename=fig_path + 'cell_locations_W_mRNA_count_q05_' + str(i) + '.' \
                                + export_args['plot_extension'])

            if verbose:
                print(p)
    except Exception as e:
        print('Some error in plotting `mod.plot_factor_spatial()`\n ' + str(e))

    if verbose:
        print('### Done ### - time ' + res_dict['run_time'])

    if return_all:
        return res_dict
    else:
        del res_dict
        del mod
        gc.collect()
        return str((time.time() - start) / 60) + ' min'
