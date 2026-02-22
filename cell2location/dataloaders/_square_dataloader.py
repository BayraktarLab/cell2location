import copy
import logging
from itertools import cycle, islice
from typing import List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import AnnTorchDataset
from scvi.dataloaders._data_splitting import validate_data_split
from torch.utils.data import DataLoader

from cell2location.dataloaders._defined_grid_dataloader import DistributedSampler

logger = logging.getLogger(__name__)


def _randomise_batches(data_iter):
    idx = torch.randperm(len(data_iter)).numpy()
    return data_iter[idx]


class SpatialGridBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Custom torch Sampler that returns a list of indices of size batch_size.
    Parameters
    ----------
    indices
        list of indices to sample from
    batch_size
        batch size of each iteration
    shuffle
        if ``True``, shuffles indices before sampling
    drop_last
        if int, drops the last batch if its length is less than drop_last.
        if drop_last == True, drops last non-full batch.
        if drop_last == False, iterate over all batches.
    """

    def __init__(
        self,
        batch_size: int = 1,
        obs_indices: np.ndarray = None,
        tile_size: int = (100, 100),
        tile_overlap: int = 10,
        positions: np.ndarray = None,
        samples: np.ndarray = None,
        shuffle: bool = True,
        drop_last: Union[bool, int] = False,
    ):
        self.batch_size = batch_size

        self.obs_indices = obs_indices
        self.n_obs = len(obs_indices)

        self.samples = samples
        self.positions = positions
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        sample_index = np.unique(samples)
        self.x_indices_per_sample = list()
        self.y_indices_per_sample = list()
        self.n_obs_per_sample = list()
        self.n_batches_per_sample = list()
        for sample in sample_index:
            x_indices_per_sample = obs_indices[samples == sample]
            positions_per_chr = positions[samples == sample]
            x_indices_per_sample = x_indices_per_sample[np.argsort(positions_per_chr)]
            self.x_indices_per_sample.append(x_indices_per_sample)
            self.n_obs_per_sample.append(len(x_indices_per_sample))
            self.n_batches_per_sample.append(int(np.ceil(len(x_indices_per_sample) / (tile_size - tile_overlap))))
        self.indices_per_sample = np.array(self.indices_per_sample, dtype=object)
        self.n_obs_per_sample = np.array(self.n_obs_per_sample)
        self.n_batches_per_sample = np.array(self.n_batches_per_sample)
        self.n_tiles = np.sum(self.n_batches_per_sample)

        self.shuffle = shuffle

        # drop last WHAT?
        last_batch_len = self.n_cells % self.batch_size
        if (drop_last is True) or (last_batch_len < drop_last):
            drop_last_n = last_batch_len
        elif (drop_last is False) or (last_batch_len >= drop_last):
            drop_last_n = 0
        else:
            raise ValueError("Invalid input for drop_last param. Must be bool or int.")
        self.drop_last_n = drop_last_n

    def get_dna_segment_batches(self):
        # start a list of DNA region batches
        dna_batches = np.array([], dtype=object)
        for chromosome in range(len(self.indices_per_chromosome)):
            # get indices of DNA regions for this chromosome (includes genes)
            indices = self.indices_per_chromosome[chromosome]
            n_vars = self.n_vars_per_chromosome[chromosome]
            # Randomly pick start index for each epoch
            if self.dna_window_random_shift is not None:
                start = torch.randperm(self.dna_window_random_shift).numpy()[0]
            else:
                start = 0
            # create region indices by splitting the region index vector into chunks
            batch_start_indices = np.arange(
                start,
                n_vars,
                step=int(self.n_vars_per_dna_region_batch - self.n_vars_overlap),
            )
            # make sure last batch is has many genes and the same length as the other batches
            if self.n_vars_overlap > 0:
                last = [
                    n_vars - self.n_vars_per_dna_region_batch - self.n_vars_overlap,
                    n_vars - self.n_vars_per_dna_region_batch,
                ]
                batch_start_indices = np.concatenate(
                    [
                        batch_start_indices[:-2],
                        last,
                    ]
                )
            else:
                batch_start_indices = np.concatenate(
                    [
                        batch_start_indices[:-1],
                        [n_vars - self.n_vars_per_dna_region_batch],
                    ]
                )
            if start > 0:
                batch_start_indices = np.concatenate(
                    [
                        [0],
                        batch_start_indices,
                    ]
                )
            # create batch indices
            batches = np.empty(len(batch_start_indices), dtype=object)
            batches[:] = [
                np.array(indices[i : i + self.n_vars_per_dna_region_batch])
                # check that the number of regions is always the same
                if len(np.array(indices[i : i + self.n_vars_per_dna_region_batch])) == self.n_vars_per_dna_region_batch
                else None
                for i in batch_start_indices
            ]  # n_batches_per_chromosome[chromosome]
            if np.any([i is None for i in batches]):
                raise ValueError("Number of regions in batch is not equal to n_vars_per_dna_region_batch")
            # append the indices to the list DNA region batches
            dna_batches = np.concatenate([dna_batches, batches])
        return dna_batches

    def get_cell_batches(self):
        """Get batches of cells.

        Returns
        -------

        """

        if self.shuffle is True:
            cells_idx = torch.randperm(self.n_cells).numpy()
        else:
            cells_idx = torch.arange(self.n_cells).numpy()

        if self.drop_last_n != 0:
            cells_idx = cells_idx[: -self.drop_last_n]

        n_cells = len(cells_idx)
        if self.n_batches_override is not None:
            # create a smaller number of batches for testing because the nested loop below is slow
            n_cells = min(n_cells, int(self.n_batches_override * self.batch_size))
        batch_start_indices = np.arange(0, n_cells, step=self.batch_size)
        cell_batches = np.empty(len(batch_start_indices), dtype=object)
        cell_batches[:] = [
            np.array(self.cell_indices[cells_idx[c : c + self.batch_size]], dtype=object) for c in batch_start_indices
        ]  # n_batches

        return cell_batches

    @staticmethod
    def apply_independent_multi_gpu_merge(batches, n_gpus):
        # merge batches from multiple GPUs
        if n_gpus > 0:
            new_size = int(np.floor(batches.shape[0] / n_gpus))
            new_batches = np.empty(new_size, dtype=object)
            for i in range(0, new_size):
                new_batches[i] = np.concatenate(batches[i * n_gpus : i * n_gpus + n_gpus])
            return new_batches
        else:
            return batches

    @staticmethod
    def apply_paired_multi_gpu_merge(data_iter, n_gpus):
        # merge batches from multiple GPUs
        if n_gpus > 0:
            new_size = int(np.floor(len(data_iter) / n_gpus))
            new_data_iter = np.empty(new_size, dtype=object)
            for i in range(0, new_size):
                new_data_iter[i] = np.array(
                    [
                        np.concatenate([i[0] for i in data_iter[i * n_gpus : i * n_gpus + n_gpus]]),
                        np.concatenate([i[1] for i in data_iter[i * n_gpus : i * n_gpus + n_gpus]]),
                    ],
                    dtype=object,
                )
            return new_data_iter
        else:
            return data_iter

    def fully_random_iter(self):
        if getattr(self, "indices_per_chromosome", None) is not None:
            dna_region_batches = self.get_dna_segment_batches()
        else:
            dna_region_batches = self.get_gene_batches()
        # if shuffle, randomise the order of batches
        if self.shuffle is True:
            dna_region_batches = _randomise_batches(dna_region_batches)
        if self.n_batches_override:
            dna_region_batches = dna_region_batches[: self.n_batches_override]
        if self.gpu_split_cells == 0:
            dna_region_batches = self.apply_independent_multi_gpu_merge(
                dna_region_batches, n_gpus=self.gpu_split_dna_windows
            )
        cell_batches = self.get_cell_batches()
        if self.gpu_split_dna_windows == 0:
            cell_batches = self.apply_independent_multi_gpu_merge(cell_batches, n_gpus=self.gpu_split_cells)
        n_cell_batches = len(cell_batches)
        n_dna_region_batches = len(dna_region_batches)
        data_iter = np.empty(n_cell_batches * n_dna_region_batches, dtype=object)
        data_iter[:] = [
            np.array(
                [np.array(cell_batches[i], dtype=int), np.array(g, dtype=int)],
                dtype=object,
            )
            for i in range(cell_batches.shape[0])
            for g in dna_region_batches
            # make sure all batches > 0 cells or genes
            if (len(cell_batches[i]) > 0) and (len(g) > 0)
        ]
        if self.shuffle and self.randomise_batches:
            data_iter = _randomise_batches(data_iter)
        data_iter = self.apply_paired_multi_gpu_merge(
            data_iter, n_gpus=min(self.gpu_split_dna_windows, self.gpu_split_cells)
        )
        return iter(data_iter)

    def get_per_label_batches(
        self,
        dna_segments,
        n_similar_labels_per_batch: int = 0,
        n_different_labels_per_batch: int = 0,
    ):
        main_label = make_selected_regions_df(  # noqa: F821
            label_averages=self.label_averages,
            dna_segments=dna_segments,
            gene_indices=self.gene_indices,
            proportion_specificity=0.02 * self.epoch_size_scaling_factor,
            proportion_absolute=0.01 * self.epoch_size_scaling_factor,
            proportion_genes_scale=2.0 * self.epoch_size_scaling_factor,
        )

        labels = np.empty(main_label.shape[0], dtype=object)
        labels[:] = [
            np.array(
                [
                    np.array(np.array([main_label.iloc[i, :]["labels"]]), dtype=object),
                    np.array(main_label.iloc[i, :]["windows"], dtype=int),
                ],
                dtype=object,
            )
            for i in range(main_label.shape[0])
        ]

        if (n_similar_labels_per_batch > 0) or (n_different_labels_per_batch > 0):
            dist = compute_ct_ct_covariance(self.label_averages)  # noqa: F821
            for i in range(main_label.shape[0]):
                extended_labels = pick_cell_types(  # noqa: F821
                    dist,
                    ind=np.where(self.label_averages.columns == labels[i][0][0])[0],
                    label_averages=self.label_averages,
                    n_similar=n_similar_labels_per_batch,
                    n_diffent=n_different_labels_per_batch,
                )
                labels[i][0] = np.concatenate([labels[i][0], extended_labels])
            assert len(labels[i][0]) == (n_similar_labels_per_batch + n_different_labels_per_batch + 1)

        return labels

    def setup_cell_by_label_cycle(self):
        if self.shuffle and self.randomise_batches:
            cells_idx = torch.randperm(self.n_cells).numpy()
        else:
            cells_idx = torch.arange(self.n_cells).numpy()

        cell_indices = self.cell_indices[cells_idx]
        labels = self.labels[cell_indices]

        self.cells_per_label_cycle = {
            label: cycle(cell_indices[np.where(labels == label)[0]]) for label in np.unique(labels)
        }

    def get_cell_by_label(self, label_batch, n_cells):
        n_cells_per_label = int(n_cells // len(label_batch))
        selected_cells = np.concatenate(
            [list(islice(self.cells_per_label_cycle[label], n_cells_per_label)) for label in label_batch]
        )
        if self.shuffle and self.randomise_batches:
            np.random.shuffle(selected_cells)

        return selected_cells

    def balanced_by_label_iter(self):
        if getattr(self, "indices_per_chromosome", None) is not None:
            dna_region_batches = self.get_dna_segment_batches()
        else:
            dna_region_batches = self.get_gene_batches()
        # get paired dna_region_batches * cell_label batches
        label_batches = self.get_per_label_batches(
            dna_segments=dna_region_batches,
            n_similar_labels_per_batch=self.n_similar_labels_per_batch,
            n_different_labels_per_batch=self.n_different_labels_per_batch,
        )
        if self.n_batches_override:
            label_batches = label_batches[: self.n_batches_override]
        # get paired dna_region_batches * cell batches
        self.setup_cell_by_label_cycle()
        data_iter = np.empty(label_batches.shape[0], dtype=object)
        for i in range(len(data_iter)):
            data_iter[i] = np.array(
                [
                    np.array(
                        self.get_cell_by_label(label_batches[i][0], n_cells=self.batch_size),
                        dtype=int,
                    ),
                    np.array(dna_region_batches[label_batches[i][1]], dtype=int),
                ],
                dtype=object,
            )
        assert (
            data_iter[i][0].shape[0] == self.batch_size
        ), "get_cell_by_label failed to provide batches of specified size"
        fetched = np.sort(np.unique(self.labels[data_iter[i][0]]))
        required = np.sort(np.unique(label_batches[i][0]))
        assert "".join(fetched) == "".join(
            required
        ), f"Broken order of label indices: fetched labels {fetched}, required labels {required}"
        logger.info(f"Number of batches: {len(data_iter)}")
        if self.shuffle and self.randomise_batches:
            data_iter = _randomise_batches(data_iter)
        data_iter = self.apply_paired_multi_gpu_merge(
            data_iter,
            n_gpus=min(self.gpu_split_dna_windows, self.gpu_split_cells),
        )
        if min(self.gpu_split_dna_windows, self.gpu_split_cells) > 0:
            logger.info(f"Number of batches (multi-GPU): {len(data_iter)}")
        self.balanced_batch_length = len(data_iter)
        return iter(data_iter)

    def get_label_batches(
        self,
        cell_batches,
        n_similar_labels_per_batch: int = 0,
        n_different_labels_per_batch: int = 0,
    ):
        # n_label_batches = (
        #                      (self.n_cells / len(self.label_averages.columns)) / cell_batches.shape[0]
        #                  ) * len(self.label_averages.columns)
        n_batches = 0  # int(max(cell_batches.shape[0], int(n_label_batches), len(self.label_averages.columns)))
        n_batches_all = len(self.label_averages.columns)
        label_batches = np.empty(n_batches + n_batches_all, dtype=object)
        label_batches[:] = [
            np.array(pd.Series([i]), dtype=object)
            for i in np.random.choice(
                self.label_averages.columns,
                size=n_batches_all,
                replace=False,
            )
            # for i in np.concatenate(
            #    [
            #        np.random.choice(
            #            self.label_averages.columns, size=n_batches_all, replace=False,
            #        ),
            #        np.random.choice(
            #            self.label_averages.columns, size=n_batches, replace=True,
            #        )
            #    ]
            # )
        ]

        if (n_similar_labels_per_batch > 0) or (n_different_labels_per_batch > 0):
            dist = compute_ct_ct_covariance(self.label_averages)  # noqa: F821
            for i in range(label_batches.shape[0]):
                extended_labels = pick_cell_types(  # noqa: F821
                    dist,
                    ind=np.where(self.label_averages.columns == label_batches[i][0])[0],
                    label_averages=self.label_averages,
                    n_similar=n_similar_labels_per_batch,
                    n_diffent=n_different_labels_per_batch,
                )
                label_batches[i] = np.concatenate([label_batches[i], extended_labels])
            assert len(label_batches[i]) == (n_similar_labels_per_batch + n_different_labels_per_batch + 1)

        return label_batches

    def balanced_label_only_iter(self):
        if getattr(self, "indices_per_chromosome", None) is not None:
            dna_region_batches = self.get_dna_segment_batches()
        else:
            dna_region_batches = self.get_gene_batches()
        # if shuffle, randomise the order of batches
        if self.shuffle is True:
            dna_region_batches = _randomise_batches(dna_region_batches)
        if self.n_batches_override:
            dna_region_batches = dna_region_batches[: self.n_batches_override]
        if self.gpu_split_cells == 0:
            dna_region_batches = self.apply_independent_multi_gpu_merge(
                dna_region_batches, n_gpus=self.gpu_split_dna_windows
            )
        # get paired dna_region_batches * cell batches
        cell_batches = self.get_cell_batches()
        label_batches = self.get_label_batches(
            cell_batches,
            n_similar_labels_per_batch=self.n_similar_labels_per_batch,
            n_different_labels_per_batch=self.n_different_labels_per_batch,
        )
        self.setup_cell_by_label_cycle()
        cell_batches = np.empty(label_batches.shape[0], dtype=object)
        for i in range(len(cell_batches)):
            cell_batches[i] = np.array(
                self.get_cell_by_label(label_batches[i], n_cells=self.batch_size),
                dtype=int,
            )
        assert (
            cell_batches[i].shape[0] == self.batch_size
        ), "get_cell_by_label failed to provide batches of specified size"
        fetched = np.sort(np.unique(self.labels[cell_batches[i]]))
        required = np.sort(np.unique(label_batches[i]))
        assert "".join(fetched) == "".join(
            required
        ), f"Broken order of label indices: fetched labels {fetched}, required labels {required}"
        if self.gpu_split_dna_windows == 0:
            cell_batches = self.apply_independent_multi_gpu_merge(cell_batches, n_gpus=self.gpu_split_cells)
        n_cell_batches = len(cell_batches)
        n_dna_region_batches = len(dna_region_batches)
        data_iter = np.empty(n_cell_batches * n_dna_region_batches, dtype=object)
        data_iter[:] = [
            np.array(
                [np.array(cell_batches[i], dtype=int), np.array(g, dtype=int)],
                dtype=object,
            )
            for i in range(cell_batches.shape[0])
            for g in dna_region_batches
            # make sure all batches > 0 cells or genes
            if (len(cell_batches[i]) > 0) and (len(g) > 0)
        ]
        if self.shuffle and self.randomise_batches:
            data_iter = _randomise_batches(data_iter)
        data_iter = self.apply_paired_multi_gpu_merge(
            data_iter, n_gpus=min(self.gpu_split_dna_windows, self.gpu_split_cells)
        )
        return iter(data_iter)

    def __iter__(self):
        return getattr(self, self.iter_type)()

    def _set_balanced_batch_length(self):
        if getattr(self, "indices_per_chromosome", None) is not None:
            dna_region_batches = self.get_dna_segment_batches()
        else:
            dna_region_batches = self.get_gene_batches()
        # get paired dna_region_batches * cell_label batches
        label_batches = self.get_per_label_batches(
            dna_segments=dna_region_batches,
            n_similar_labels_per_batch=self.n_similar_labels_per_batch,
            n_different_labels_per_batch=self.n_different_labels_per_batch,
        )
        balanced_batch_length = len(label_batches)
        if (self.gpu_split_cells > 0) and (self.gpu_split_dna_windows > 0):
            balanced_batch_length = balanced_batch_length // self.gpu_split_cells
        elif self.gpu_split_cells > 0:
            balanced_batch_length = balanced_batch_length // self.gpu_split_cells

        self.balanced_batch_length = balanced_batch_length

    def __len__(self):
        if self.n_batches_override is not None:
            return self.n_batches_override * self.n_batches_override

        if self.iter_type == "balanced_by_label_iter":
            return self.balanced_batch_length

        from math import ceil

        if self.drop_last_n != 0:
            n_dna_region_batches = self.n_genes // self.gene_batch_size
            n_cells_batches = self.n_cells // self.batch_size

            if self.labels is not None:
                # n_label_batches = (
                #                      (self.n_cells / len(self.label_averages.columns)) / n_cells_batches
                #                  ) * len(self.label_averages.columns)
                # n_cells_batches = int(max(
                #    n_cells_batches,
                #    int(n_label_batches),
                #    len(self.label_averages.columns)
                # ))
                n_cells_batches = 0
                n_cells_batches = n_cells_batches + len(self.label_averages.columns)

            if self.gpu_split_cells > 0:
                n_cells_batches = n_cells_batches // self.gpu_split_cells
            if self.gpu_split_dna_windows > 0:
                n_dna_region_batches = n_dna_region_batches // self.gpu_split_dna_windows
            length = n_cells_batches * n_dna_region_batches
        else:
            n_dna_region_batches = ceil(self.n_genes / self.gene_batch_size)
            n_cells_batches = ceil(self.n_cells / self.batch_size)

            if self.labels is not None:
                # n_label_batches = (
                #                      (self.n_cells / len(self.label_averages.columns)) / n_cells_batches
                #                  ) * len(self.label_averages.columns)
                # n_cells_batches = int(max(
                #    n_cells_batches,
                #    int(n_label_batches),
                #    len(self.label_averages.columns)
                # ))
                n_cells_batches = 0
                n_cells_batches = n_cells_batches + len(self.label_averages.columns)

            if self.gpu_split_cells > 0:
                n_cells_batches = n_cells_batches // self.gpu_split_cells
            if self.gpu_split_dna_windows > 0:
                n_dna_region_batches = n_dna_region_batches // self.gpu_split_dna_windows
            length = n_cells_batches * n_dna_region_batches
        return length


class DistributedBatchSampler(PerGeneChromatinBatchSampler):  # noqa: F821
    """`BatchSampler` wrapper that distributes across each batch multiple workers. Copied from PyTorch NLP.

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.

    Example:
        >>> from torch.utils.data.sampler import BatchSampler
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]
    """

    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(DistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)


class PerGeneChromatinAnnDataLoader(DataLoader):
    """
    DataLoader for loading tensors from AnnData objects.
    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object with a registered AnnData object.
    shuffle
        Whether the data should be shuffled
    indices
        The indices of the observations in the adata to load
    batch_size
        minibatch size to load each iteration
    data_and_attributes
        Dictionary with keys representing keys in data registry (``adata_manager.data_registry``)
        and value equal to desired numpy loading type (later made into torch tensor).
        If ``None``, defaults to all registered data.
    data_loader_kwargs
        Keyword arguments for :class:`~torch.utils.data.DataLoader`
    iter_ndarray
        Whether to iterate over numpy arrays instead of torch tensors
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        gene_bool: List[bool],
        gene_region_coo: coo_matrix = None,
        chromosomes: np.ndarray = None,
        positions: np.ndarray = None,
        shuffle: bool = True,
        full_indices: np.ndarray = None,
        cell_indices=None,
        gene_indices=None,
        region_indices=None,
        local_indices=None,
        filter_tensors: dict = None,
        n_vars_per_dna_region_batch: int = 420,
        n_vars_overlap: int = 0,
        dna_window_random_shift: int = None,
        cell_plate_inputs: List[str] = None,
        var_plate_inputs: List[str] = None,
        square_var_plate_inputs: List[str] = None,
        per_site_plate_inputs: List[str] = None,
        batch_size: int = 128,
        gene_batch_size: int = 128,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        iter_ndarray: bool = False,
        n_batches_override: Optional[int] = None,
        n_regions_override: Optional[int] = None,
        randomise_batches: bool = True,
        return_gene_indices: bool = True,
        use_ddp: bool = False,
        n_sites: int = 50,
        gpu_split_cells: int = 0,
        gpu_split_dna_windows: int = 0,
        window_label_stratification: str = None,
        window_label_stratification_layer: str = None,
        iter_type: str = "fully_random_iter",
        n_similar_labels_per_batch: int = 0,
        n_different_labels_per_batch: int = 0,
        epoch_size_scaling_factor: float = 1.0,
        **data_loader_kwargs,
    ):
        if adata_manager.adata is None:
            raise ValueError("Please run register_fields() on your AnnDataManager object first.")

        if data_and_attributes is not None:
            data_registry = adata_manager.data_registry
            for key in data_and_attributes.keys():
                if key not in data_registry.keys():
                    raise ValueError(f"{key} required for model but not registered with AnnDataManager.")

        self.dataset = AnnTorchDataset(
            adata_manager,
            cell_plate_inputs=cell_plate_inputs,
            var_plate_inputs=var_plate_inputs,
            square_var_plate_inputs=square_var_plate_inputs,
            per_site_plate_inputs=per_site_plate_inputs,
            getitem_tensors=data_and_attributes,
            filter_tensors=filter_tensors,
            filter_by="gene_bool",
            n_sites=n_sites,
        )
        # print(self.dataset[[[100, 53, 1], [0, 5, 6]]])

        # compute per label average for all variables
        labels = label_averages = None
        if window_label_stratification is not None:
            from cell2location.cluster_averages.cluster_averages import (
                compute_cluster_averages,
            )

            label_averages = compute_cluster_averages(
                adata_manager.adata,
                labels=window_label_stratification,
                layer=window_label_stratification_layer,
            )
            labels = adata_manager.adata.obs[window_label_stratification].values

        sampler_kwargs = {
            "gene_region_coo": gene_region_coo,
            "chromosomes": chromosomes,
            "positions": positions,
            "labels": labels,
            "label_averages": label_averages,
            "batch_size": batch_size,
            "gene_batch_size": gene_batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "n_batches_override": n_batches_override,
            "n_regions_override": n_regions_override,
            "randomise_batches": randomise_batches,
            "return_gene_indices": return_gene_indices,
            "n_vars_per_dna_region_batch": n_vars_per_dna_region_batch,
            "n_vars_overlap": n_vars_overlap,
            "dna_window_random_shift": dna_window_random_shift,
            "gpu_split_cells": gpu_split_cells,
            "gpu_split_dna_windows": gpu_split_dna_windows,
            "iter_type": iter_type,
            "n_similar_labels_per_batch": n_similar_labels_per_batch,
            "n_different_labels_per_batch": n_different_labels_per_batch,
            "epoch_size_scaling_factor": epoch_size_scaling_factor,
        }

        if cell_indices is None:
            indices = np.arange(adata_manager.adata.n_obs)
            sampler_kwargs["cell_indices"] = indices
        else:
            if hasattr(cell_indices, "dtype") and cell_indices.dtype is np.dtype("bool"):
                cell_indices = np.where(cell_indices)[0].ravel()
            cell_indices = np.asarray(cell_indices)
            sampler_kwargs["cell_indices"] = cell_indices

        if full_indices is None:
            indices = np.arange(adata_manager.adata.n_vars)
            sampler_kwargs["full_indices"] = indices
        else:
            sampler_kwargs["full_indices"] = indices = full_indices
        if gene_indices is None:
            gene_indices = indices[gene_bool]
            sampler_kwargs["gene_indices"] = gene_indices
        else:
            if hasattr(gene_indices, "dtype") and gene_indices.dtype is np.dtype("bool"):
                gene_indices = np.where(gene_indices)[0].ravel()
            gene_indices = np.asarray(gene_indices)
            sampler_kwargs["gene_indices"] = gene_indices

        if region_indices is None:
            region_indices = indices[~gene_bool]
            sampler_kwargs["region_indices"] = region_indices
        else:
            if hasattr(region_indices, "dtype") and region_indices.dtype is np.dtype("bool"):
                region_indices = np.where(region_indices)[0].ravel()
            region_indices = np.asarray(region_indices)
            sampler_kwargs["region_indices"] = region_indices

        if local_indices is None:
            local_indices = list(np.arange(len(gene_indices))) + list(np.arange(len(region_indices)))
        sampler_kwargs["local_indices"] = local_indices

        if gene_region_coo is not None:
            if gene_region_coo.shape[0] != len(local_indices[gene_bool]):
                raise ValueError(
                    f"Number of genes in gene_region_coo {gene_region_coo.shape[0]} must match "
                    f"the number of genes in local_indices {len(local_indices[gene_bool])}"
                )
            if not (
                (gene_region_coo.shape[1] == len(local_indices[~gene_bool]))
                or (gene_region_coo.shape[1] == len(local_indices))
            ):
                raise ValueError(
                    f"Number of regions in gene_region_coo {gene_region_coo.shape[1]} must match "
                    f"the number of regions {len(local_indices[~gene_bool])} in local_indices"
                )
        if positions is not None:
            if chromosomes is None:
                raise ValueError("If positions are provided, chromosomes must also be provided.")
            if len(positions) != len(full_indices):
                raise ValueError(
                    f"Number of positions {len(positions)} must match "
                    f"the number of variables adata.n_vars {len(full_indices)}"
                )
            if len(chromosomes) != len(full_indices):
                raise ValueError(
                    f"Number of positions in `chromosomes` {len(chromosomes)} must match "
                    f"the number of variables adata.n_vars {len(full_indices)}"
                )

        self.cell_indices = cell_indices
        self.gene_indices = gene_indices
        self.region_indices = region_indices
        self.local_indices = local_indices
        self.sampler_kwargs = sampler_kwargs
        self.gpu_split_dna_windows = gpu_split_dna_windows
        self.gpu_split_cells = gpu_split_cells
        sampler = PerGeneChromatinBatchSampler(**self.sampler_kwargs)  # noqa: F821
        if use_ddp:
            sampler = DistributedBatchSampler(
                sampler,
                gpu_split_cells=True if gpu_split_cells > 0 else False,
                gpu_split_dna_windows=True if gpu_split_dna_windows > 0 else False,
            )
            # logger.warning(f"sampler {[*sampler.__iter__()][0:2]}")
        # logger.warning(f"isinstance(sampler, torch.utils.data.RandomSampler) : {isinstance(sampler, torch.utils.data.RandomSampler)}")
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        # do not touch batch size here, sampler gives batched indices
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": None})

        if iter_ndarray:
            self.data_loader_kwargs.update({"collate_fn": _dummy_collate})

        super().__init__(self.dataset, **self.data_loader_kwargs)


def _dummy_collate(b):
    """Dummy collate to have dataloader return numpy ndarrays."""
    return b


class PerGeneChromatinDataSplitter(pl.LightningDataModule):
    """
    Creates data loaders ``train_set``, ``validation_set``, ``test_set``.
    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.
    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    use_gpu
        Use default GPU if available (if None or True), or index of GPU to use (if int),
        or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
    **kwargs
        Keyword args for data loader. If adata has labeled data, data loader
        class is :class:`~scvi.dataloaders.SemiSupervisedDataLoader`,
        else data loader class is :class:`~scvi.dataloaders.AnnDataLoader`.
    Examples
    --------
    >>> adata = scvi.data.synthetic_iid()
    >>> scvi.model.SCVI.setup_anndata(adata)
    >>> adata_manager = scvi.model.SCVI(adata).adata_manager
    >>> splitter = DataSplitter(adata)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        gene_bool,
        train_size: float = 1.0,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
        use_ddp: bool = False,
        shuffle_training: bool = True,
        drop_last: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.gene_bool = gene_bool
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.data_loader_kwargs = kwargs
        self.data_loader_kwargs["gene_bool"] = gene_bool
        self.use_gpu = use_gpu
        self.use_ddp = use_ddp
        self.shuffle_training = shuffle_training
        self.drop_last = drop_last

        self.n_train_ = dict()
        self.n_val_ = dict()
        if self.data_loader_kwargs.get("chromosomes", None) is not None:
            self.n_train_["n_genes"] = n_genes = 100  # this is ignored
            self.n_val_["n_genes"] = 0  # this is ignored
        else:
            n_genes = np.sum(gene_bool)
            self.n_train_["n_genes"], self.n_val_["n_genes"] = validate_data_split(
                n_genes,
                self.train_size,
                self.validation_size,
            )
        self.n_train_["n_obs"], self.n_val_["n_obs"] = validate_data_split(
            self.adata_manager.adata.n_obs,
            self.train_size,
            self.validation_size,
        )

    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""
        n_train = self.n_train_["n_obs"]
        n_val = self.n_val_["n_obs"]
        random_state = np.random.RandomState(seed=settings.seed)

        cell_indices = np.arange(self.adata_manager.adata.n_obs)
        cell_indices = random_state.permutation(cell_indices)
        self.cell_indices_val_idx = cell_indices[:n_val]
        self.cell_indices_train_idx = cell_indices[n_val : (n_val + n_train)]
        self.cell_indices_test_idx = cell_indices[(n_val + n_train) :]

        self.local_indices = self.adata_manager.get_from_registry(registry_key="local_indices").flatten()

        n_train = self.n_train_["n_genes"]
        n_val = self.n_val_["n_genes"]
        indices = np.arange(self.adata_manager.adata.n_vars)
        self.full_indices = indices
        self.region_indices = indices[~self.gene_bool]

        gene_indices = indices[self.gene_bool]
        gene_indices = random_state.permutation(gene_indices)
        self.val_idx = gene_indices[:n_val]
        self.train_idx = gene_indices[n_val : (n_val + n_train)]
        self.test_idx = gene_indices[(n_val + n_train) :]

        # accelerator, _, self.device = parse_use_gpu_arg(
        #    self.use_gpu, return_device=True
        # )
        # self.pin_memory = (
        #    True
        #    if (settings.dl_pin_memory_gpu_training and accelerator == "gpu")
        #    else False
        # )
        self.pin_memory = True if (settings.dl_pin_memory_gpu_training and self.use_gpu) else False

    def train_dataloader(self):
        return PerGeneChromatinAnnDataLoader(
            self.adata_manager,
            full_indices=self.full_indices,
            cell_indices=self.cell_indices_train_idx,
            gene_indices=self.train_idx,
            region_indices=self.region_indices,
            local_indices=self.local_indices,
            shuffle=self.shuffle_training,
            randomise_batches=self.shuffle_training,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            use_ddp=self.use_ddp,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            return PerGeneChromatinAnnDataLoader(
                self.adata_manager,
                full_indices=self.full_indices,
                cell_indices=self.cell_indices_val_idx,
                gene_indices=self.val_idx,
                region_indices=self.region_indices,
                local_indices=self.local_indices,
                shuffle=False,
                randomise_batches=False,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return PerGeneChromatinAnnDataLoader(
                self.adata_manager,
                full_indices=self.full_indices,
                cell_indices=self.cell_indices_test_idx,
                gene_indices=self.test_idx,
                region_indices=self.region_indices,
                local_indices=self.local_indices,
                shuffle=False,
                randomise_batches=False,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
