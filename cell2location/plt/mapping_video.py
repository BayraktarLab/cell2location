# +
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Circle
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec


def get_rgb_function(cmap, min_value, max_value):
    r"""Generate a function to map continous values to RGB values using colormap between min_value & max_value."""

    if min_value > max_value:
        raise ValueError("Max_value should be greater or than min_value.")

    if min_value == max_value:
        warnings.warn(
            "Max_color is equal to min_color. It might be because of the data or bad parameter choice. "
            "If you are using plot_contours function try increasing max_color_quantile parameter and"
            "removing cell types with all zero values."
        )

        def func_equal(x):
            factor = 0 if max_value == 0 else 0.5
            return cmap(np.ones_like(x) * factor)

        return func_equal

    def func(x):
        return cmap((np.clip(x, min_value, max_value) - min_value) / (max_value - min_value))

    return func


def rgb_to_ryb(rgb):
    """
    Converts colours from RGB colorspace to RYB

    Parameters
    ----------

    rgb
        numpy array Nx3

    Returns
    -------
    Numpy array Nx3
    """
    rgb = np.array(rgb)
    if len(rgb.shape) == 1:
        rgb = rgb[np.newaxis, :]

    white = rgb.min(axis=1)
    black = (1 - rgb).min(axis=1)
    rgb = rgb - white[:, np.newaxis]

    yellow = rgb[:, :2].min(axis=1)
    ryb = np.zeros_like(rgb)
    ryb[:, 0] = rgb[:, 0] - yellow
    ryb[:, 1] = (yellow + rgb[:, 1]) / 2
    ryb[:, 2] = (rgb[:, 2] + rgb[:, 1] - yellow) / 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = ryb[mask].max(axis=1) / rgb[mask].max(axis=1)
        ryb[mask] = ryb[mask] / norm[:, np.newaxis]

    return ryb + black[:, np.newaxis]


def ryb_to_rgb(ryb):
    """
    Converts colours from RYB colorspace to RGB

    Parameters
    ----------

    ryb
        numpy array Nx3

    Returns
    -------
    Numpy array Nx3
    """
    ryb = np.array(ryb)
    if len(ryb.shape) == 1:
        ryb = ryb[np.newaxis, :]

    black = ryb.min(axis=1)
    white = (1 - ryb).min(axis=1)
    ryb = ryb - black[:, np.newaxis]

    green = ryb[:, 1:].min(axis=1)
    rgb = np.zeros_like(ryb)
    rgb[:, 0] = ryb[:, 0] + ryb[:, 1] - green
    rgb[:, 1] = green + ryb[:, 1]
    rgb[:, 2] = (ryb[:, 2] - green) * 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = rgb[mask].max(axis=1) / ryb[mask].max(axis=1)
        rgb[mask] = rgb[mask] / norm[:, np.newaxis]

    return rgb + white[:, np.newaxis]


def plot_spatial(
    spot_factors_df,
    coords,
    labels,
    text=None,
    circle_diameter=4.0,
    alpha_scaling=1.0,
    max_col=(np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
    max_color_quantile=0.98,
    show_img=True,
    img=None,
    img_alpha=1.0,
    adjust_text=False,
    plt_axis="off",
    axis_y_flipped=True,
    x_y_labels=("", ""),
    crop_x=None,
    crop_y=None,
    text_box_alpha=0.9,
    reorder_cmap=range(7),
    style="fast",
    colorbar_position="bottom",
    colorbar_label_kw={},
    colorbar_shape={},
    colorbar_tick_size=12,
    colorbar_grid=None,
    image_cmap="Greys_r",
    white_spacing=20,
):
    r"""Plot spatial abundance of cell types (regulatory programmes) with colour gradient and interpolation.
      This method supports only 7 cell types with these colours (in order, which can be changed using reorder_cmap).
      'yellow' 'orange' 'blue' 'green' 'purple' 'grey' 'white'
    :param spot_factors_df: pd.DataFrame - spot locations of cell types, only 6 cell types allowed
    :param coords: np.ndarray - x and y coordinates (in columns) to be used for ploting spots
    :param text: pd.DataFrame - with x, y coordinates, text to be printed
    :param circle_diameter: diameter of circles
    :param labels: list of strings, labels of cell types
    :param alpha_scaling: adjust color alpha
    :param max_col: crops the colorscale maximum value for each column in spot_factors_df.
    :param max_color_quantile: crops the colorscale at x quantile of the data.
    :param show_img: show image?
    :param img: numpy array representing a tissue image.
        If not provided a black background image is used.
    :param img_alpha: transparency of the image
    :param lim: x and y max limits on the plot. Minimum is always set to 0, if `lim` is None maximum
        is set to image height and width. If 'no_limit' then no limit is set.
    :param adjust_text: move text label to prevent overlap
    :param plt_axis: show axes?
    :param axis_y_flipped: flip y axis to match coordinates of the plotted image
    :param reorder_cmap: reorder colors to make sure you get the right color for each category

    :param style: plot style (matplolib.style.context):
        'fast' - white background & dark text;
        'dark_background' - black background & white text;

    :param colorbar_position: 'bottom', 'right' or None
    :param colorbar_label_kw: dict that will be forwarded to ax.set_label()
    :param colorbar_shape: dict {'vertical_gaps': 1.5, 'horizontal_gaps': 1.5,
                                    'width': 0.2, 'height': 0.2}, not obligatory to contain all params
    :param colorbar_tick_size: colorbar ticks label size
    :param colorbar_grid: tuple of colorbar grid (rows, columns)
    :param image_cmap: matplotlib colormap for grayscale image
    :param white_spacing: percent of colorbars to be hidden

    """

    # TODO add parameter description

    if spot_factors_df.shape[1] > 7:
        raise ValueError("Maximum of 7 cell types / factors can be plotted at the moment")

    def create_colormap(R, G, B):
        spacing = int(white_spacing * 2.55)

        N = 255
        M = 3

        alphas = np.concatenate([[0] * spacing * M, np.linspace(0, 1.0, (N - spacing) * M)])

        vals = np.ones((N * M, 4))
        #         vals[:, 0] = np.linspace(1, R / 255, N * M)
        #         vals[:, 1] = np.linspace(1, G / 255, N * M)
        #         vals[:, 2] = np.linspace(1, B / 255, N * M)
        for i, color in enumerate([R, G, B]):
            vals[:, i] = color / 255
        vals[:, 3] = alphas

        return ListedColormap(vals)

    # Create linearly scaled colormaps
    YellowCM = create_colormap(240, 228, 66)  # #F0E442 ['#F0E442', '#D55E00', '#56B4E9',
    # '#009E73', '#5A14A5', '#C8C8C8', '#323232']
    RedCM = create_colormap(213, 94, 0)  # #D55E00
    BlueCM = create_colormap(86, 180, 233)  # #56B4E9
    GreenCM = create_colormap(0, 158, 115)  # #009E73
    GreyCM = create_colormap(200, 200, 200)  # #C8C8C8
    WhiteCM = create_colormap(50, 50, 50)  # #323232
    PurpleCM = create_colormap(90, 20, 165)  # #5A14A5

    cmaps = [YellowCM, RedCM, BlueCM, GreenCM, PurpleCM, GreyCM, WhiteCM]

    cmaps = [cmaps[i] for i in reorder_cmap]

    with mpl.style.context(style):

        fig = plt.figure()

        if colorbar_position == "right":

            if colorbar_grid is None:
                colorbar_grid = (len(labels), 1)

            shape = {"vertical_gaps": 1.5, "horizontal_gaps": 0, "width": 0.15, "height": 0.2}
            shape = {**shape, **colorbar_shape}

            gs = GridSpec(
                nrows=colorbar_grid[0] + 2,
                ncols=colorbar_grid[1] + 1,
                width_ratios=[1, *[shape["width"]] * colorbar_grid[1]],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0], 1],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )
            ax = fig.add_subplot(gs[:, 0], aspect="equal", rasterized=True)

        if colorbar_position == "bottom":
            if colorbar_grid is None:
                if len(labels) <= 3:
                    colorbar_grid = (1, len(labels))
                else:
                    n_rows = round(len(labels) / 3 + 0.5 - 1e-9)
                    colorbar_grid = (n_rows, 3)

            shape = {"vertical_gaps": 0.3, "horizontal_gaps": 0.6, "width": 0.2, "height": 0.035}
            shape = {**shape, **colorbar_shape}

            gs = GridSpec(
                nrows=colorbar_grid[0] + 1,
                ncols=colorbar_grid[1] + 2,
                width_ratios=[0.3, *[shape["width"]] * colorbar_grid[1], 0.3],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0]],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )

            ax = fig.add_subplot(gs[0, :], aspect="equal", rasterized=True)

        if colorbar_position is None:
            ax = fig.add_subplot(aspect="equal", rasterized=True)

        if colorbar_position is not None:
            cbar_axes = []
            for row in range(1, colorbar_grid[0] + 1):
                for column in range(1, colorbar_grid[1] + 1):
                    cbar_axes.append(fig.add_subplot(gs[row, column]))

            n_excess = colorbar_grid[0] * colorbar_grid[1] - len(labels)
            if n_excess > 0:
                for i in range(1, n_excess + 1):
                    cbar_axes[-i].set_visible(False)

        ax.set_xlabel(x_y_labels[0])
        ax.set_ylabel(x_y_labels[1])

        if img is not None and show_img:
            ax.imshow(img, aspect="equal", alpha=img_alpha, origin="lower", cmap=image_cmap)

        # crop images in needed
        if crop_x is not None:
            ax.set_xlim(crop_x[0], crop_x[1])
        if crop_y is not None:
            ax.set_ylim(crop_y[0], crop_y[1])

        if axis_y_flipped:
            ax.invert_yaxis()

        if plt_axis == "off":
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        # pick spot weights from just one sample
        counts = spot_factors_df.values.copy()

        # plot spots as circles
        c_ord = list(np.arange(0, counts.shape[1]))

        colors = np.zeros((*counts.shape, 4))
        weights = np.zeros(counts.shape)

        for c in c_ord:

            min_color_intensity = counts[:, c].min()
            max_color_intensity = np.min([np.quantile(counts[:, c], max_color_quantile), max_col[c]])

            rgb_function = get_rgb_function(cmap=cmaps[c], min_value=min_color_intensity, max_value=max_color_intensity)

            color = rgb_function(counts[:, c])
            color[:, 3] = color[:, 3] * alpha_scaling

            norm = mpl.colors.Normalize(vmin=min_color_intensity, vmax=max_color_intensity)

            if colorbar_position is not None:
                cbar_ticks = [
                    min_color_intensity,
                    np.mean([min_color_intensity, max_color_intensity]),
                    max_color_intensity,
                ]
                cbar_ticks = np.array(cbar_ticks)

                if max_color_intensity > 13:
                    cbar_ticks = cbar_ticks.astype(np.int32)
                else:
                    cbar_ticks = cbar_ticks.round(2)

                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmaps[c]),
                    cax=cbar_axes[c],
                    orientation="horizontal",
                    extend="both",
                    ticks=cbar_ticks,
                )

                cbar.ax.tick_params(labelsize=colorbar_tick_size)
                max_color = rgb_function(max_color_intensity / 1.5)
                cbar.ax.set_title(labels[c], **{**{"size": 20, "color": max_color, "alpha": 1}, **colorbar_label_kw})

            colors[:, c] = color
            weights[:, c] = np.clip(counts[:, c] / (max_color_intensity + 1e-10), 0, 1)
            weights[:, c][counts[:, c] < min_color_intensity] = 0

        colors_ryb = np.zeros((*weights.shape, 3))

        for i in range(colors.shape[0]):
            colors_ryb[i] = rgb_to_ryb(colors[i, :, :3])

        def kernel(w):
            return w ** 2

        kernel_weights = kernel(weights[:, :, np.newaxis])
        weighted_colors_ryb = (colors_ryb * kernel_weights).sum(axis=1) / kernel_weights.sum(axis=1)

        weighted_colors = np.zeros((weights.shape[0], 4))

        weighted_colors[:, :3] = ryb_to_rgb(weighted_colors_ryb)

        weighted_colors[:, 3] = colors[:, :, 3].max(axis=1)

        ax.scatter(x=coords[:, 0], y=coords[:, 1], c=weighted_colors, s=circle_diameter ** 2)

        # add text
        if text is not None:
            bbox_props = dict(boxstyle="round", ec="0.5", alpha=text_box_alpha, fc="w")
            texts = []
            for x, y, s in zip(
                np.array(text.iloc[:, 0].values).flatten(),
                np.array(text.iloc[:, 1].values).flatten(),
                text.iloc[:, 2].tolist(),
            ):
                texts.append(ax.text(x, y, s, ha="center", va="bottom", bbox=bbox_props))

            if adjust_text:
                from adjustText import adjust_text

                adjust_text(texts, arrowprops=dict(arrowstyle="->", color="w", lw=0.5))

    return fig


def interpolate_coord(start=10, end=5, steps=100, accel_power=3, accelerate=True, jitter=None):
    r"""Interpolate coordinates between start_array and end_array positions in N steps
        with non-linearity in movement according to acc_power,
        and accelerate change in coordinates (True) or slow it down (False).
    :param jitter: shift positions by a random number by sampling:
                  new_coord = np.random.normal(mean=coord, sd=jitter), reasonable values 0.01-0.1
    """

    seq = np.linspace(np.zeros_like(start), np.ones_like(end), steps)
    seq = seq ** accel_power

    if jitter is not None:
        seq = np.random.normal(loc=seq, scale=jitter * np.abs(seq))
        seq[0] = np.zeros_like(start)
        seq[steps - 1] = np.ones_like(end)

    if accelerate:
        seq = 1 - seq
        start

    seq = seq * (start - end) + end
    if not accelerate:
        seq = np.flip(seq, axis=0)

    return seq


def expand_1by1(df):
    col6 = [df.copy() for i in range(df.shape[1])]
    index = df.index.astype(str)
    columns = df.columns
    for i in range(len(col6)):
        col6_1 = col6[i]

        col6_1_new = np.zeros_like(col6_1)
        col6_1_new[:, i] = col6_1[col6_1.columns[i]].values

        col6_1_new = pd.DataFrame(col6_1_new, index=index + str(i), columns=columns)
        col6[i] = col6_1_new

    return pd.concat(col6, axis=0)


def plot_video_mapping(
    adata_vis,
    adata,
    sample_ids,
    spot_factors_df,
    sel_clust,
    sel_clust_col,
    sample_id,
    sc_img=None,
    sp_img=None,
    sp_img_scaling_fac=1,
    adata_cluster_col="annotation_1",
    cell_fact_df=None,
    step_n=[20, 100, 15, 45, 80, 30],
    step_quantile=[1, 1, 1, 1, 0.95, 0.95],
    sc_point_size=1,
    aver_point_size=20,
    sp_point_size=5,
    reorder_cmap=range(7),
    label_clusters=False,
    style="dark_background",
    adjust_text=False,
    sc_alpha=0.6,
    sp_alpha=0.8,
    img_alpha=0.8,
    sc_power=20,
    sp_power=20,
    sc_accel_power=3,
    sp_accel_power=3,
    sc_accel_decel=True,
    sp_accel_decel=False,
    sc_jitter=None,
    sp_jitter=None,
    save_path="./results/mouse_viseum_snrna/std_model/mapping_video/",
    crop_x=None,
    crop_y=None,
    save_extension="png",
    colorbar_shape={"vertical_gaps": 2, "horizontal_gaps": 0.13},
):
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
    coords = adata_vis.obsm["spatial"].copy() * sp_img_scaling_fac

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
        img_alpha_seq = interpolate_coord(
            start=0, end=img_alpha, steps=step_n[3] + 1, accel_power=sc_power, accelerate=True, jitter=None
        )

    # extract umap coordinates
    umap_coord = adata.obsm["X_umap"].copy()

    # make positive and rescale to fill the image
    umap_coord[:, 0] = umap_coord[:, 0] + abs(umap_coord[:, 0].min()) + abs(umap_coord[:, 0].max()) * 0.01
    umap_coord[:, 1] = -umap_coord[:, 1]  # flip y axis
    umap_coord[:, 1] = umap_coord[:, 1] + abs(umap_coord[:, 1].min()) + abs(umap_coord[:, 1].max()) * 0.01

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
    cell_fact_df["other"] = (cell_fact_df.sum(1) == 0).astype(np.int64)

    # compute average position weighted by cell density
    aver_coord = pd.DataFrame()
    for c in cell_fact_df.columns:
        dens = cell_fact_df[c].values
        dens = dens / dens.sum(0)
        aver = np.array((umap_coord * dens.reshape((cell_fact_df.shape[0], 1))).sum(0))
        aver_coord_1 = pd.DataFrame(aver.reshape((1, 2)), index=[c], columns=["x", "y"])
        aver_coord_1["column"] = c
        aver_coord = pd.concat([aver_coord, aver_coord_1])

    aver_coord = aver_coord.loc[aver_coord.index != "other"]

    # compute movement of cells toward averages (increasing size)
    moving_averages1 = [
        interpolate_coord(
            start=umap_coord,
            end=np.ones_like(umap_coord) * aver_coord.loc[i, ["x", "y"]].values,
            steps=step_n[1] + 1,
            accel_power=sc_accel_power,
            accelerate=sc_accel_decel,
            jitter=sc_jitter,
        )
        for i in aver_coord.index
    ]
    moving_averages1 = np.array(moving_averages1)

    # (increasing dot size) for cells -> averages
    circ_diam1 = interpolate_coord(
        start=sc_point_size,
        end=aver_point_size,
        steps=step_n[1] + 1,
        accel_power=sc_power,
        accelerate=sc_accel_decel,
        jitter=None,
    )

    # compute movement of spots from averages to locations
    moving_averages2 = [
        interpolate_coord(
            start=np.ones_like(sel_coords) * aver_coord.loc[i, ["x", "y"]].values,
            end=sel_coords,
            steps=step_n[4] + 1,
            accel_power=sp_accel_power,
            accelerate=sp_accel_decel,
            jitter=sp_jitter,
        )
        for i in aver_coord.index
    ]
    moving_averages2 = np.array(moving_averages2)

    # (decreasing dot size) for averages -> locations
    circ_diam2 = interpolate_coord(
        start=aver_point_size,
        end=sp_point_size,
        steps=step_n[4] + 1,
        accel_power=sp_power,
        accelerate=sp_accel_decel,
        jitter=None,
    )

    #### start saving plots ####
    # plot UMAP with no changes
    for i0 in tqdm(range(step_n[0])):
        fig = plot_spatial(
            cell_fact_df,
            coords=umap_coord,
            labels=cell_fact_df.columns,
            circle_diameter=sc_point_size,
            alpha_scaling=sc_alpha,
            img=sc_img,
            img_alpha=1,
            style=style,
            # determine max color level using data quantiles
            max_color_quantile=step_quantile[0],  # set to 1 to pick max - essential for discrete scaling
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + 1}.{save_extension}", bbox_inches="tight")
        fig.clear()

    # plot evolving UMAP from cells to averages
    for i1 in tqdm(range(step_n[1])):
        ann_no_other = cell_fact_df[cell_fact_df.columns[cell_fact_df.columns != "other"]]
        ann_no_other = expand_1by1(ann_no_other)
        coord = np.concatenate(moving_averages1[:, i1, :, :], axis=0)

        fig = plot_spatial(
            ann_no_other,
            coords=coord,
            labels=ann_no_other.columns,
            circle_diameter=circ_diam1[i1],
            alpha_scaling=sc_alpha,
            img=sc_img,
            img_alpha=1,
            style=style,
            # determine max color level using data quantiles
            max_color_quantile=step_quantile[1],  # set to 1 to pick max - essential for discrete scaling
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + i1 + 2}.{save_extension}", bbox_inches="tight")
        fig.clear()

    # plot averages
    if label_clusters:
        label_clusters = aver_coord[["x", "y", "column"]]
    else:
        label_clusters = None
    for i2 in tqdm(range(step_n[2])):
        ann_no_other = cell_fact_df[cell_fact_df.columns[cell_fact_df.columns != "other"]]
        ann_no_other = expand_1by1(ann_no_other)
        coord = np.concatenate(moving_averages1[:, i1 + 1, :, :], axis=0)

        fig = plot_spatial(
            ann_no_other,
            coords=coord,
            labels=ann_no_other.columns,
            text=label_clusters,
            circle_diameter=circ_diam1[i1 + 1],
            alpha_scaling=sc_alpha,
            img=sc_img,
            img_alpha=1,
            style=style,
            # determine max color level using data quantiles
            max_color_quantile=step_quantile[2],  # set to 1 to pick max - essential for discrete scaling
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + i1 + i2 + 3}.{save_extension}", bbox_inches="tight")
        fig.clear()

    # plot averages & fade-in histology image
    for i22 in tqdm(range(step_n[3])):
        ann_no_other = cell_fact_df[cell_fact_df.columns[cell_fact_df.columns != "other"]]
        ann_no_other = expand_1by1(ann_no_other)
        coord = np.concatenate(moving_averages1[:, i1 + 1, :, :], axis=0)

        fig = plot_spatial(
            ann_no_other,
            coords=coord,
            labels=ann_no_other.columns,
            text=label_clusters,
            circle_diameter=circ_diam1[i1 + 1],
            alpha_scaling=sc_alpha,
            img=sp_img,
            img_alpha=img_alpha_seq[i22],
            style=style,
            # determine max color level using data quantiles
            max_color_quantile=step_quantile[3],  # set to 1 to pick max - essential for discrete scaling
            adjust_text=adjust_text,
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + i1 + i2 + i22 + 4}.{save_extension}", bbox_inches="tight")
        fig.clear()

    # plot evolving from averages to spatial locations
    for i3 in tqdm(range(step_n[4])):

        # sel_clust_df_1 = expand_1by1(sel_clust_df)

        dfs = []
        clusters = []
        for i in range(sel_clust_df.shape[1]):
            idx = sel_clust_df.values.argmax(axis=1) == i
            dfs.append(moving_averages2[i, i3, idx, :])
            clusters.append(sel_clust_df[idx])
        # coord = moving_averages2[0, i3, :, :]
        coord = np.concatenate(dfs, axis=0)

        fig = plot_spatial(
            pd.concat(clusters, axis=0),
            coords=coord,
            labels=sel_clust_df.columns,
            circle_diameter=circ_diam2[i3],
            alpha_scaling=sp_alpha,
            img=sp_img,
            img_alpha=img_alpha,
            style=style,
            max_color_quantile=step_quantile[4],
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + i1 + i2 + i2 + i3 + 5}.{save_extension}", bbox_inches="tight")
        fig.clear()

    # plot a few final images
    for i4 in tqdm(range(step_n[5])):

        dfs = []
        clusters = []
        for i in range(sel_clust_df.shape[1]):
            idx = sel_clust_df.values.argmax(axis=1) == i
            dfs.append(moving_averages2[i, i3 + 1, idx, :])
            clusters.append(sel_clust_df[idx])
        # coord = moving_averages2[0, i3, :, :]
        for d in dfs:
            print(d.shape)
        coord = np.concatenate(dfs, axis=0)

        fig = plot_spatial(
            pd.concat(clusters, axis=0),
            coords=coord,
            labels=sel_clust_df.columns,
            circle_diameter=circ_diam2[i3 + 1],
            alpha_scaling=sp_alpha,
            img=sp_img,
            img_alpha=img_alpha,
            style=style,
            max_color_quantile=step_quantile[5],
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + i1 + i2 + i2 + i3 + i4 + 6}.{save_extension}", bbox_inches="tight")
        fig.clear()
