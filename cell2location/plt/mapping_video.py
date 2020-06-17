# +
#from matplotlib.collections import PatchCollection
#from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def get_rgb_function(cmap, min_value, max_value):
    r""" Generate a function to map continous values to RGB values using colormap between min_value & max_value.
    """
    def func(x):
        return cmap((np.clip(x, min_value, max_value)-min_value)/(max_value-min_value))
    return func

def plot_contours(spot_factors_df, coords, text=None,
                  circle_diameter=4, alpha_scaling=0.6,
                  col_breaks=[0.1, 100, 1000, 3000],
                  max_col=[5000, 5000, 5000, 5000, 5000, 5000, 5000],
                  max_color_quantile=0.95, 
                  show_img=True, img=None, img_alpha=1,
                  plot_contour=False,
                  save_path=None, save_name='', save_facecolor='black',
                  show_fig=True, lim=None,
                  fontsize=12, adjust_text=False,
                  plt_axis='off', axis_y_flipped=True, x_y_labels=['', ''],
                  crop_x=None, crop_y=None, text_box_alpha=0.9,
                  reorder_cmap=range(7), overwrite_color=None):
    r"""
    :param spot_factors_df: pd.DataFrame - spot locations of cell types, only 6 cell types allowed
    :param coords: np.ndarray - x and y coordinates (in columns) to be used for ploting spots
    :param text: pd.DataFrame - with x, y coordinates, text to be printed
    :param circle_diameter: diameter of circles
    :param alpha_scaling: adjust color alpha
    :param col_breaks: contour plot levels
    :param max_col: crops the colorscale maximum value for each column in spot_factors_df.
    :param max_color_quantile: crops the colorscale at x quantile of the data.
    :param show_img: show image?
    :param img: numpy array representing a tissue image. 
                    If not provided a black background image is used.
    :param img_alpha: transparency of the image
    :param plot_contour: boolean, whether to plot contours (not implemented yet).
    :param save_path: if not None - directory where to save images, otherwise the plot is shown.
    :param save_name: file name when saving the plot
    :param show_fig: boolean, show figure?
    :param lim: x and y max limits on the plot. Minimum is always set to 0, if `lim` is None maximum 
                    is set to image height and width. If 'no_limit' then no limit is set.
    :param fontsize: text fontsize
    :param adjust_text: move text label to prevent overlap
    :param plt_axis: show axes?
    :param axis_y_flipped: flip y axis to match coordinates of the plotted image
    :param reorder_cmap: reorder colors to make sure you get the right color for each category
    """
    
    if spot_factors_df.shape[1] > 7:
        raise ValueError('Maximum of 7 cell types / factors can be plotted at the moment')
    
    alphas = np.concatenate((np.abs(np.linspace(0, 0, 256 - 200)), np.abs(np.linspace(0, 1.0, 256 - 56))))
    N = 256
    
    # Create linearly scaled colormaps
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, 240/256, N)
    vals[:, 1] = np.linspace(1, 228/256, N)
    vals[:, 2] = np.linspace(1, 66/256, N)
    vals[:, 3] = alphas
    YellowCM = ListedColormap(vals)

    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, 213/256, N)
    vals[:, 1] = np.linspace(1, 94/256, N)
    vals[:, 2] = np.linspace(1, 0/256, N)
    vals[:, 3] = alphas
    RedCM = ListedColormap(vals)

    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, 86/256, N)
    vals[:, 1] = np.linspace(1, 180/256, N)
    vals[:, 2] = np.linspace(1, 233/256, N)
    vals[:, 3] = alphas
    BlueCM = ListedColormap(vals)

    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, 0/256, N)
    vals[:, 1] = np.linspace(1, 158/256, N)
    vals[:, 2] = np.linspace(1, 115/256, N)
    vals[:, 3] = alphas
    GreenCM = ListedColormap(vals)
    
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, 200/256, N)
    vals[:, 1] = np.linspace(1, 200/256, N)
    vals[:, 2] = np.linspace(1, 200/256, N)
    vals[:, 3] = alphas
    GreyCM = ListedColormap(vals)
    
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, 50/256, N)
    vals[:, 1] = np.linspace(1, 50/256, N)
    vals[:, 2] = np.linspace(1, 50/256, N)
    vals[:, 3] = alphas
    WhiteCM = ListedColormap(vals)
    
    PurpleCM = plt.cm.get_cmap('Purples')
    PurpleCM._init()
    alphas = np.concatenate((np.abs(np.linspace(0, 0, 259 - 200)), np.abs(np.linspace(0, 1.0, 259 - 59))))
    PurpleCM._lut[:,-1] = alphas
    
    cmaps=[]
    cmaps = cmaps + [YellowCM]
    cmaps = cmaps + [RedCM]
    cmaps = cmaps + [BlueCM]
    cmaps = cmaps + [GreenCM]
    cmaps = cmaps + [PurpleCM]
    cmaps = cmaps + [GreyCM]
    cmaps = cmaps + [WhiteCM]
    
    cmaps = [cmaps[i] for i in reorder_cmap]
    
    # modify shape of alpha scaling
    alpha_scaling = alpha_scaling * np.ones_like(max_col)
    
    from matplotlib import rcParams
    fig = plt.figure(figsize=rcParams["figure.figsize"])
        
    # pick spot weights from just one sample
    weights = spot_factors_df.values.copy()
        
    # plot tissue image
    if img is not None and show_img:
        plt.imshow(img, aspect='equal', alpha=img_alpha)
        img_lim_0 = int(img.shape[1])
        img_lim_1 = int(img.shape[0])
        
    elif show_img:
        if len(coords.shape) == 3:
            xy = coords.max((0,1))
        else:
            xy = coords.max(0)
            
        img = np.zeros((int(xy[1]), int(xy[0]), 3))
        plt.imshow(img, aspect='equal', alpha=img_alpha) 
        
    else:
        if len(coords.shape) == 3:
            xy = coords.max((0,1))
        else:
            xy = coords.max(0)
            
        if axis_y_flipped:
            img_lim_0 = int(xy[1])
            img_lim_1 = int(xy[0])
        else:
            img_lim_0 = int(xy[0])
            img_lim_1 = int(xy[1])
        
    # plot spots as circles
    c_ord = list(np.arange(0, weights.shape[1])) + list(np.linspace(start=weights.shape[1]-1,
                                                                        stop=0, num=weights.shape[1],
                                                                        dtype=int))
        
    for c in c_ord:
        rgb_function = get_rgb_function(cmaps[c],
                                        weights[:,c].min(),
                                        np.min([np.quantile(weights[:,c], max_color_quantile), max_col[c]]))
            
        #for idx in range(coords_s.shape[0]):
        #    
        #    dots = [Circle((coords_s[idx, 0], coords_s[idx, 1]), circle_diameter)] # dot coordinates
        #    color = rgb_function(weights[idx, c])
        #    coll = PatchCollection(dots, facecolor=color, # dot color
        #                           alpha=color[3] * 0.5,
        #                           edgecolor=None)
        #    
        #    fig.axes[0].add_collection(coll)
        if len(coords.shape) == 3:
            coords_s = coords[c,:,:]
        else:
            coords_s = coords
            
        color = rgb_function(weights[:, c])
        
        if overwrite_color is not None:
            color=overwrite_color * alpha_scaling[c]
            
        color[:, 3] = color[:, 3] * alpha_scaling[c]
        
        plt.scatter(x=coords_s[:,0], y=coords_s[:,1],
                    c=color, s=circle_diameter**2
                    #cmap=cmaps[c],
                    #alpha=0.1
                    )
                
    # plot_contours
    if plot_contour:
        for c in range(weights.shape[1]):
            #TO DO: first, compute a 2d histogram, next plot contours
            raise ValueError('plotting contours is not implemented yet (useful for showing density in scRNA-seq)')
            CS1 = plt.contour(coords_s, weights[:,c], col_breaks, vmin=0, vmax=100, cmap=cmaps[c], 
                                alpha=1, linestyles='dashed', linewidths=2)
            plt.clabel(CS1, inline=1, fontsize=15, fmt='%1.0f')
            
    # add text
    if text is not None:
        bbox_props = dict(boxstyle="round", ec="0.5",
                          alpha=text_box_alpha, fc="w", #facecolor=PurpleCM(1)
                         )
        texts = []
        for x, y, s in zip(np.array(text.iloc[:,0].values).flatten(),
                           np.array(text.iloc[:,1].values).flatten(),
                           text.iloc[:,2].tolist()):
            texts.append(plt.text(x, y, s, 
                                  ha="center", va="bottom",
                                  bbox=bbox_props))
        
        if adjust_text:
            
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle="->", color='w', lw=0.5))
        
    plt.xlabel(x_y_labels[0])
    plt.ylabel(x_y_labels[1])
    
    if lim == 'no_limit':
        pass
    elif lim is None:
        plt.xlim(0, img_lim_0)
        if axis_y_flipped:
            plt.ylim(img_lim_1, 0)
        else:
            plt.ylim(0, img_lim_1)
    else:
        plt.xlim(0, lim[0])
        if axis_y_flipped:
            plt.ylim(lim[1], 0)
        else:
            plt.ylim(0, lim[1])
            
    plt.gca().axis(plt_axis)
    
    # crop images in needed
    if crop_x is not None:
        plt.xlim(crop_x[0], crop_x[1])
    if crop_y is not None:
        plt.ylim(crop_y[0], crop_y[1])
        
    
    if show_fig:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path + 'density_maps_{}.png'.format(save_name),
                    bbox_inches='tight', facecolor=save_facecolor)
            
    fig.clear()
    plt.close(fig)

def interpolate_coord(start = 10, end = 5, steps = 100, accel_power=3,
                      accelerate=True, jitter=None):
    r""" Interpolate coordinates between start_array and end_array positions in N steps 
    with non-linearity in movement according to acc_power, 
    and accelerate change in coordinates (True) or slow it down (False).
    :param jitter: shift positions by a random number by sampling:
                  new_coord = np.random.normal(mean=coord, sd=jitter), reasonable values 0.01-0.1
    """
    
    seq = np.linspace(np.zeros_like(start), np.ones_like(end), steps)
    seq = seq ** accel_power
    
    if jitter is not None:
        seq = np.random.normal(loc=seq, scale=jitter*np.abs(seq))
        seq[0] = np.zeros_like(start)
        seq[steps-1] = np.ones_like(end)

    if accelerate:
        seq = 1 - seq
        start
        
    seq = seq * (start - end) + end
    if not accelerate:
        seq = np.flip(seq, axis=0)
        
    return seq

def plot_video_mapping(adata_vis, adata, sample_ids, spot_factors_df,
                    sel_clust, sel_clust_col,
                    sample_id='s144600', sc_img=None, 
                    sp_img=None, sp_img_scaling_fac=1,
                    adata_cluster_col='annotation_1', cell_fact_df=None,
                    step_n=[20, 100, 15, 45, 80, 30], step_quantile=[1, 1, 1, 1, 0.95, 0.95],
                    sc_point_size=1, aver_point_size=20, sp_point_size=5,
                    fontsize=15, adjust_text=False,
                    sc_alpha=0.6, sp_alpha=0.8, img_alpha=0.8,
                    sc_power=20, sp_power=20, 
                    sc_accel_power=3, sp_accel_power=3,
                    sc_accel_decel=True, sp_accel_decel=False,
                    sc_jitter=None, sp_jitter=None,   
                    save_path='./results/mouse_viseum_snrna/std_model/mapping_video/',
                    crop_x=None, crop_y=None):
    r"""Create frames for a video illustrating the approach from UMAP of single cells to their spatial locations.
    We use linear interpolation of UMAP and spot coordinates to create movement.
    :param adata_vis: anndata with Visium data (including spatial slot in `.obsm`)
    :param adata: anndata with single cell data (including X_umap slot in `.obsm`)
    :param sample_ids: pd.Series - sample ID for each spot
    :param spot_factors_df: output of the model showing spatial expression of cell types / factors.
    :param sel_clust: selected cluster names in `adata_cluster_col` column of adata.obs
    :param sel_clust_col: selected cluster column name in spot_factors_df
    :param sample_id: sample id to use for visualisation
    :param adata_cluster_col: column in adata.obs containing cluster annotations
    :param cell_fact_df: alternative to adata_cluster_col, pd.DataFrame specifying class for each cell (can be continuous).
    :param step_n: how many frames to record in each step: UMAP, UMAP collapsing into averages, averages, averages expanding into locations, locations.
    :param step_quantile: how to choose maximum colorscale limit in each step? (quantile) Use 1 for discrete values.
    :param sc_point_size: point size for cells
    :param aver_point_size: point size for averages
    :param sp_point_size: point size for spots
    :param fontsize: size of text label of averages
    :param adjust_text: adjust text label position to avoid overlaps
    :param sc_alpha, sp_alpha: color alpha scaling for single cells and spatial.
    :param sc_power, sp_power: change dot size nonlinearly with this exponent 
    :param sc_accel_power, sp_accel_power: change movement speed size nonlinearly with this exponent 
    :param sc_accel_decel, sp_accel_decel: accelerate (True) or decelereate (False)
    :param save_path: path where to save frames (named according to order of steps)
    """
    
    from tqdm.auto import tqdm
    
    # extract spot expression and coordinates
    coords = adata_vis.obsm['spatial'].copy() * sp_img_scaling_fac
    
    s_ind = sample_ids.isin([sample_id])
    sel_clust_df = spot_factors_df.loc[s_ind, sel_clust_col]
    sel_coords = coords[s_ind, :]
    sample_id = sample_ids[s_ind]

    if sc_img is None:
        # create a black background image
        xy = sel_coords.max(0) + sel_coords.max(0) * 0.05
        sc_img = np.zeros((int(xy[1]), int(xy[0]), 3))
        
    if sp_img is None:
        # create a black background image
        xy = sel_coords.max(0) + sel_coords.max(0) * 0.05
        sp_img = np.zeros((int(xy[1]), int(xy[0]), 3))
        img_alpha = 1
        img_alpha_seq = 1
    else:
        img_alpha_seq = interpolate_coord(start = 0, end = img_alpha, steps = step_n[3]+1, 
                                          accel_power=sc_power, accelerate=True, jitter=None)
    
    # extract umap coordinates
    umap_coord = adata.obsm['X_umap'].copy()
    
    # make positive and rescale to fill the image
    umap_coord[:, 0] = umap_coord[:, 0] + abs(umap_coord[:, 0].min()) + abs(umap_coord[:, 0].max())*0.01
    umap_coord[:, 1] = -umap_coord[:, 1] # flip y axis
    umap_coord[:, 1] = umap_coord[:, 1] + abs(umap_coord[:, 1].min()) + abs(umap_coord[:, 1].max())*0.01
    
    if crop_x is None:
        img_width = sc_img.shape[0] * 0.99
        x_offset = 0
        umap_coord[:, 0] = umap_coord[:, 0] / umap_coord[:, 0].max() * img_width
    else:
        img_width = abs(crop_x[0] - crop_x[1]) * 0.99
        x_offset = np.array(crop_x).min()
        umap_coord[:, 0] = umap_coord[:, 0] / umap_coord[:, 0].max() * img_width
        umap_coord[:, 0] = umap_coord[:, 0] + x_offset
        
    if crop_y is None:
        img_height = sc_img.shape[1] * 0.99
        y_offset = 0
        y_offset2 = 0
        umap_coord[:, 1] = umap_coord[:, 1] / umap_coord[:, 1].max() * img_height
    else:
        img_height = abs(crop_y[0] - crop_y[1]) * 0.99
        y_offset = np.array(crop_y).min()
        y_offset2 = sp_img.shape[1] - np.array(crop_y).max()
        umap_coord[:, 1] = umap_coord[:, 1] / umap_coord[:, 1].max() * img_height
        umap_coord[:, 1] = umap_coord[:, 1] + y_offset

    if cell_fact_df is None:
        cell_fact_df = pd.get_dummies(adata.obs[adata_cluster_col], columns=[adata_cluster_col])
    
    cell_fact_df = cell_fact_df[sel_clust]
    cell_fact_df.columns = cell_fact_df.columns.tolist()
    cell_fact_df['other'] = (cell_fact_df.sum(1) == 0).astype(np.int64)
    
    # compute average position weighted by cell density
    aver_coord = pd.DataFrame()
    for c in cell_fact_df.columns:
        dens = cell_fact_df[c].values
        dens = dens / dens.sum(0)
        aver = np.array((umap_coord * dens.reshape((cell_fact_df.shape[0], 1))).sum(0))
        aver_coord_1 = pd.DataFrame(aver.reshape((1,2)),
                                    index=[c], columns=['x', 'y'])
        aver_coord_1['column'] = c
        aver_coord = pd.concat([aver_coord, aver_coord_1])
        
    aver_coord = aver_coord.loc[aver_coord.index != 'other']
    
    # compute movement of cells toward averages (increasing size) 
    moving_averages1 = [interpolate_coord(start = umap_coord, end = np.ones_like(umap_coord) \
                            * aver_coord.loc[i, ['x' ,'y']].values,
                      steps = step_n[1]+1, accel_power=sc_accel_power,
                      accelerate=sc_accel_decel, jitter=sc_jitter) 
                        for i in aver_coord.index]
    moving_averages1 = np.array(moving_averages1)

    # (increasing dot size) for cells -> averages
    circ_diam1 = interpolate_coord(start = sc_point_size,
                      end = aver_point_size, steps = step_n[1]+1, 
                      accel_power=sc_power, accelerate=sc_accel_decel,
                      jitter=None)
    
    # compute movement of spots from averages to locations
    moving_averages2 = [interpolate_coord(start = np.ones_like(sel_coords) \
                            * aver_coord.loc[i, ['x' ,'y']].values,
                      end = sel_coords, steps = step_n[4]+1, 
                      accel_power=sp_accel_power,
                      accelerate=sp_accel_decel, jitter=sp_jitter) 
                        for i in aver_coord.index]
    moving_averages2 = np.array(moving_averages2)
    
    # (decreasing dot size) for averages -> locations
    circ_diam2 = interpolate_coord(start = aver_point_size,
                      end = sp_point_size, steps = step_n[4]+1, 
                      accel_power=sp_power, accelerate=sp_accel_decel,
                      jitter=None)
    
    #### start saving plots ####
    # plot UMAP with no changes
    for i0 in range(step_n[0]):
        plot_contours(cell_fact_df,
                  coords=umap_coord,
                  circle_diameter=sc_point_size, alpha_scaling=sc_alpha,
                  img=sc_img, img_alpha=1, plot_contour=False,
                  # determine max color level using data quantiles
                  max_color_quantile=step_quantile[0], # set to 1 to pick max - essential for discrete scaling
                  save_path=save_path, save_name=str(i0 + 1), #axis_y_flipped=False,
                  show_fig=False, crop_x=crop_x, crop_y=crop_y)
    
    # plot evolving UMAP from cells to averages
    for i1 in tqdm(range(step_n[1])):
        ann_no_other = cell_fact_df[cell_fact_df.columns[cell_fact_df.columns != 'other']]
        plot_contours(ann_no_other,
                  coords=moving_averages1[:,i1,:,:], 
                  circle_diameter=circ_diam1[i1], alpha_scaling=sc_alpha,
                  img=sc_img, img_alpha=1, plot_contour=False,
                  # determine max color level using data quantiles
                  max_color_quantile=step_quantile[1], # set to 1 to pick max - essential for discrete scaling
                  save_path=save_path, save_name=str(i0 + i1 + 2), #axis_y_flipped=False,
                  show_fig=False, crop_x=crop_x, crop_y=crop_y)
        
    # plot averages
    for i2 in range(step_n[2]):
        ann_no_other = cell_fact_df[cell_fact_df.columns[cell_fact_df.columns != 'other']]
        plot_contours(ann_no_other,
                  coords=moving_averages1[:,i1 + 1,:,:], 
                  text=aver_coord[['x' ,'y', 'column']],
                  circle_diameter=circ_diam1[i1 + 1], alpha_scaling=sc_alpha,
                  img=sc_img, img_alpha=1, plot_contour=False,
                  # determine max color level using data quantiles
                  max_color_quantile=step_quantile[2], # set to 1 to pick max - essential for discrete scaling
                  save_path=save_path, save_name=str(i0 + i1 + i2 + 3), #axis_y_flipped=False,
                  show_fig=False, fontsize=fontsize,
                  adjust_text=adjust_text, crop_x=crop_x, crop_y=crop_y)
        
    # plot averages & fade-in histology image
    for i22 in range(step_n[3]):
        ann_no_other = cell_fact_df[cell_fact_df.columns[cell_fact_df.columns != 'other']]
        plot_contours(ann_no_other,
                  coords=moving_averages1[:,i1 + 1,:,:], 
                  text=aver_coord[['x' ,'y', 'column']],
                  circle_diameter=circ_diam1[i1 + 1], alpha_scaling=sc_alpha,
                  img=sp_img, img_alpha=img_alpha_seq[i22], plot_contour=False,
                  # determine max color level using data quantiles
                  max_color_quantile=step_quantile[3], # set to 1 to pick max - essential for discrete scaling
                  save_path=save_path, save_name=str(i0 + i1 + i2 + i22 + 4), 
                  show_fig=False, fontsize=fontsize,
                  adjust_text=adjust_text, crop_x=crop_x, crop_y=crop_y)
    
    # plot evolving UMAP from cells to averages
    for i3 in tqdm(range(step_n[4])):
        plot_contours(sel_clust_df,
                  coords=moving_averages2[:,i3,:,:],
                  circle_diameter=circ_diam2[i3], alpha_scaling=sp_alpha,
                  img=sp_img, img_alpha=img_alpha, plot_contour=False,
                  max_color_quantile=step_quantile[4],
                  save_path=save_path, save_name=str(i0 + i1 + i2 + i2 + i3 + 5),
                  show_fig=False, crop_x=crop_x, crop_y=crop_y)
    
    # plot a few final images
    for i4 in range(step_n[5]):
        plot_contours(sel_clust_df,
                  coords=moving_averages2[:,i3+1,:,:], 
                  circle_diameter=circ_diam2[i3+1],
                  alpha_scaling=sp_alpha,
                  img=sp_img, img_alpha=img_alpha, plot_contour=False,
                  max_color_quantile=step_quantile[5],
                  save_path=save_path, save_name=str(i0 + i1 + i2 + i2 + i3 + i4 + 6),
                  show_fig=False, crop_x=crop_x, crop_y=crop_y)

