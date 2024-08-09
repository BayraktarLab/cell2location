# import geopandas as gpd
import math
from copy import copy

# from shapely.geometry import Polygon
# from geopandas import GeoDataFrame
from typing import Iterator, Optional, TypeVar, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import scvi
import torch
import torch.distributed as dist
from scipy.sparse import csc_matrix
from scvi.data import AnnDataManager
from scvi.dataloaders import AnnTorchDataset
from scvi.dataloaders._data_splitting import validate_data_split
from torch.utils.data import DataLoader
from torch.utils.data.distributed import Sampler

__all__ = [
    "DistributedSampler",
]

T_co = TypeVar("T_co", covariant=True)


def assign_tiles_to_locations(
    adata,
    rows: int,
    cols: int,
    spatial_key: str = "spatial",
    batch_key: str = None,
):
    """
    Create a grid of tiles and assign each location to a tile.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates in `adata.obsm[spatial_key]`.
    rows
        Number of rows in the grid.
    cols
        Number of columns in the grid.
    spatial_key
        Key in `adata.obsm` where the spatial coordinates are stored.
    batch_key
        Key in `adata.obs` where the batch information is stored. Tiles are created for each batch separately.

    Returns

    -------
    AnnData object with a new column in `adata.obs` called "tile" that contains the tile index for each location.
    The object is sorted by tile index.
    """
    if batch_key is None:
        adata.obs["batch"] = "0"
        batch_key = "batch"
    adata.obs[f"{spatial_key}_x"] = adata.obsm[spatial_key][:, 0]
    adata.obs[f"{spatial_key}_y"] = adata.obsm[spatial_key][:, 1]
    if "tiles" in adata.obs.columns:
        adata.obs["tiles"] = ""
    for batch in adata.obs[batch_key].unique():
        adata_batch = adata[adata.obs[batch_key] == batch, :].copy()
        x_start_positions = np.arange(
            np.min(adata_batch.obsm[spatial_key][:, 0]), np.max(adata_batch.obsm[spatial_key][:, 0]), step=rows
        )
        y_start_positions = np.arange(
            np.min(adata_batch.obsm[spatial_key][:, 1]), np.max(adata_batch.obsm[spatial_key][:, 1]), step=cols
        )
        ind_x = np.digitize(adata_batch.obsm[spatial_key][:, 0], x_start_positions)
        ind_y = np.digitize(adata_batch.obsm[spatial_key][:, 1], y_start_positions)
        adata.obs.loc[adata.obs[batch_key] == batch, "tiles"] = (
            adata.obs[batch_key].astype(str)
            + pd.Series("_", index=adata.obs_names).astype(str)
            + pd.Series(ind_x, index=adata.obs_names).astype(str)
            + pd.Series("_", index=adata.obs_names).astype(str)
            + pd.Series(ind_y, index=adata.obs_names).astype(str)
        )
    sorting_index = adata.obs.sort_values(by=["tiles", f"{spatial_key}_x", f"{spatial_key}_y"]).index
    adata = adata[sorting_index, :].copy()
    adata.obsm["tiles"] = csc_matrix(pd.get_dummies(adata.obs["tiles"], sparse=True).values.astype("uint32"))
    adata.uns["tiles_names"] = np.array(pd.get_dummies(adata.obs["tiles"], sparse=True).columns.values.astype("str"))
    return adata


def expand_tiles(
    adata_vis,
    tile_key: str = "leiden",
    distance: float = 2000.0,
    distance_step: float = 100.0,
    threshold: float = 0.001,
    overlap: float = 2.0,
    distances_key: str = "distances",
):
    current_overlap = 0.0
    while current_overlap < overlap:
        from scipy.sparse import csr_matrix

        distances = adata_vis.obsp[distances_key].copy()
        distances.data[distances.data >= distance] = 0
        expanded = distances.astype("float32") @ csr_matrix(pd.get_dummies(adata_vis.obs[tile_key]).values).astype(
            "float32"
        )
        expanded = pd.DataFrame(
            expanded.toarray(),
            index=adata_vis.obs_names,
            columns=pd.get_dummies(adata_vis.obs[tile_key]).columns,
        )
        expanded = expanded > threshold
        if current_overlap == expanded.sum(1).mean():
            break
        current_overlap = expanded.sum(1).mean()
        distance = distance + distance_step
    return expanded, pd.get_dummies(adata_vis.obs[tile_key])


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
        indices: np.ndarray = None,
        tiles: csc_matrix = None,
        shuffle: bool = True,
        drop_last: Union[bool, int] = False,
    ):
        self.batch_size = batch_size

        self.indices = indices
        self.n_obs = len(indices)

        self.tiles = tiles.astype("bool")
        self.tiles_index = np.arange(tiles.shape[1]).astype("uint32")
        self.n_tiles = tiles.shape[1]

        self.shuffle = shuffle

        # drop last WHAT?
        last_batch_len = self.n_tiles % self.batch_size
        if (drop_last is True) or (last_batch_len < drop_last):
            drop_last_n = last_batch_len
        elif (drop_last is False) or (last_batch_len >= drop_last):
            drop_last_n = 0
        else:
            raise ValueError("Invalid input for drop_last param. Must be bool or int.")
        self.drop_last_n = drop_last_n

    def get_tile_batches(self):
        """Get batches of tiles.

        Returns
        -------
        Iterable over batches of tiles.

        """

        if self.shuffle is True:
            tile_idx = torch.randperm(self.n_tiles).numpy()
        else:
            tile_idx = torch.arange(self.n_tiles).numpy()

        if self.drop_last_n != 0:
            tile_idx = tile_idx[: -self.drop_last_n]

        n_tiles = len(tile_idx)
        batch_start_indices = np.arange(0, n_tiles, step=self.batch_size)
        tile_batches = np.empty(len(batch_start_indices), dtype=object)
        tile_batches[:] = [
            np.array(self.tiles_index[tile_idx[c : c + self.batch_size]], dtype=object) for c in batch_start_indices
        ]  # n_batches

        return tile_batches

    def get_obs_batches(self, tile_batches):
        """
        Get batches of observations.

        Returns
        -------
        Iterable over batches of observations.

        """
        obs_batches = np.empty(len(tile_batches), dtype=object)
        obs_batches[:] = [
            np.array(self.indices[np.asarray(self.tiles[:, tiles].sum(1)).flatten().astype("bool")], dtype="int64")
            for tiles in tile_batches
        ]
        return obs_batches

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

    def __iter__(self):
        tile_batches = self.get_tile_batches()
        obs_batches = self.get_obs_batches(tile_batches)
        return iter(obs_batches)

    def __len__(self):
        from math import ceil

        if self.drop_last_n != 0:
            n_batches = self.n_tiles // self.batch_size
        else:
            n_batches = ceil(self.n_tiles / self.batch_size)
        return n_batches


class DistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True``, sampler will shuffle the
            indices. Default is ``False`` because shuffling within a batch is irrelevant.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``True``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        iterable,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )
        self.iterable = iterable
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # TODO pick the correct sizes for total cells and genes in the current minibatch
        # distributed sampler should only distribute cell indices not gene indices
        self.total_size_tiles = len(self.iterable)
        if self.drop_last and (self.total_size_tiles % self.num_replicas) != 0:  # type: ignore[arg-type]
            # Split to the nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_tile_samples = math.ceil(
                (self.total_size_tiles - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_tile_samples = math.ceil(self.total_size_tiles / self.num_replicas)  # type: ignore[arg-type]
        self.total_tile_size = self.num_tile_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        tile_indices = self.iterable

        if not self.drop_last:
            raise NotImplementedError("DistributedSampler with drop_last=False is not implemented yet")
        else:
            # remove tail of data to make it evenly divisible.
            tile_indices = tile_indices[: self.total_tile_size]
        assert len(tile_indices) == self.total_tile_size

        # subsample
        items_per_gpu = int(self.total_tile_size / self.num_replicas)
        tile_indices = tile_indices[self.rank * items_per_gpu : (self.rank + 1) * items_per_gpu]
        assert len(tile_indices) == self.num_tile_samples
        return iter(tile_indices)

    def __len__(self) -> int:
        return self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DistributedBatchSampler(SpatialGridBatchSampler):
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
        for batch in iter(self.batch_sampler.get_tile_batches()):
            yield self.batch_sampler.get_obs_batches(list(DistributedSampler(batch, **self.kwargs)))

    def __len__(self):
        return int(len(self.batch_sampler) / self.kwargs["num_replicas"])


class SpatialGridAnnDataLoader(DataLoader):
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
        indices: np.ndarray = None,
        # tiles: np.ndarray = None,
        shuffle: bool = True,
        batch_size: int = 1,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        iter_ndarray: bool = False,
        use_ddp: bool = False,
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
            getitem_tensors=data_and_attributes,
        )
        # print(self.dataset[[[100, 53, 1], [0, 5, 6]]])

        sampler_kwargs = {
            "tiles": adata_manager.get_from_registry("tiles"),
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
        }

        if indices is None:
            indices = np.arange(adata_manager.adata.n_obs).astype("int64")
            sampler_kwargs["indices"] = indices
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices).astype("int64")
            sampler_kwargs["indices"] = indices

        self.sampler_kwargs = sampler_kwargs
        sampler = SpatialGridBatchSampler(**self.sampler_kwargs)
        if use_ddp:
            sampler = DistributedBatchSampler(
                sampler,
            )
        self.data_loader_kwargs = copy(data_loader_kwargs)
        # do not touch batch size here, sampler gives batched indices
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": None})

        if iter_ndarray:
            self.data_loader_kwargs.update({"collate_fn": _dummy_collate})

        super().__init__(self.dataset, **self.data_loader_kwargs)


def _dummy_collate(b):
    """Dummy collate to have dataloader return numpy ndarrays."""
    return b


class SpatialGridDataSplitter(pl.LightningDataModule):
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
        train_size: float = 1.0,
        validation_size: Optional[float] = None,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        use_ddp: bool = False,
        shuffle_training: bool = True,
        drop_last: bool = False,
        pin_memory: bool = False,
        shuffle_set_split: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.data_loader_kwargs = kwargs
        self.accelerator = accelerator
        self.device = device
        self.use_ddp = use_ddp
        self.shuffle_training = shuffle_training
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.shuffle_set_split = shuffle_set_split

        self.n_train_ = dict()
        self.n_val_ = dict()
        # if self.data_loader_kwargs.get("tiles", None) is None:
        #    raise ValueError("tiles must be specified in data_loader_kwargs")
        # tiles = self.data_loader_kwargs.get("tiles", None)
        tiles = self.adata_manager.get_from_registry("tiles")
        n_tiles = tiles.shape[1]
        self.n_train_["n_tiles"], self.n_val_["n_tiles"] = validate_data_split(
            n_tiles,
            self.train_size,
            self.validation_size,
        )

    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""
        n_train = self.n_train_["n_tiles"]
        n_val = self.n_val_["n_tiles"]
        random_state = np.random.RandomState(seed=scvi.settings.seed)

        tiles = self.adata_manager.get_from_registry("tiles")
        # tiles = self.data_loader_kwargs.get("tiles", None)
        tiles_index = np.arange(tiles.shape[1])
        n_tiles = tiles.shape[1]

        tile_idx = np.arange(n_tiles)
        if self.shuffle_set_split:
            tile_idx = random_state.permutation(tile_idx)

        self.tile_idx_train_idx = tiles_index[tile_idx[:n_train]]
        self.tile_idx_val_idx = tiles_index[tile_idx[n_train : (n_val + n_train)]]
        self.tile_idx_test_idx = tiles_index[tile_idx[(n_val + n_train) :]]

        obs_idx = np.arange(self.adata_manager.adata.n_obs)

        self.val_idx = obs_idx[np.asarray(tiles[:, self.tile_idx_val_idx].sum(1)).ravel().astype("bool")]
        self.train_idx = obs_idx[np.asarray(tiles[:, self.tile_idx_train_idx].sum(1)).ravel().astype("bool")]
        self.test_idx = obs_idx[np.asarray(tiles[:, self.tile_idx_test_idx].sum(1)).ravel().astype("bool")]

        self.pin_memory = True if (self.pin_memory and self.accelerator == "gpu") else False

    def train_dataloader(self):
        return SpatialGridAnnDataLoader(
            self.adata_manager,
            shuffle=self.shuffle_training,
            indices=self.train_idx,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            use_ddp=self.use_ddp,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            return SpatialGridAnnDataLoader(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=False,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return SpatialGridAnnDataLoader(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=False,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
