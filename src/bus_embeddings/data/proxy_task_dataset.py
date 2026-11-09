"""Proxy-task observations: one sample per observed travel or dwell time.

Each link observation yields a travel time target relative to the ideal
(free-flow) travel time of the link. Each stop observation yields a stop
label (dwell time > 0) and, if the bus stopped, a dwell time target. Targets
are z-scored with scalers fitted on the training data.
"""
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import holidays
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bus_embeddings.logging import get_logger

logger = get_logger(__name__)

_NL_HOLIDAYS = holidays.Netherlands(years=range(2015, 2030))  # type: ignore[attr-defined]

# Time context: sin/cos time of day, day-of-week one-hot (7), sin/cos month
# of year, public holiday flag.
TIME_CONTEXT_DIM = 12

# Rows handled at once by compute_time_context; only affects peak memory.
TIME_CONTEXT_CHUNK_ROWS = 200_000

# One batch: (link_indices, link_ctx, link_targets,
#             stop_indices, stop_ctx, stop_targets, stop_labels)
Batch = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    Optional[torch.Tensor],
]


# Observation columns used after preprocessing (to_time, from_stop and
# to_stop are not needed for training or scoring)
TRAVEL_COLUMNS = ["from_time", "travel_time", "link", "municipality"]
DWELL_COLUMNS = ["stop", "from_time", "travel_time", "municipality"]


def load_observations(
    travel_times_parquet: str | Path,
    dwell_times_parquet: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the preprocessed travel and dwell time observations (UTC timestamps).

    Reads only the columns that training and scoring use (see
    ``TRAVEL_COLUMNS`` and ``DWELL_COLUMNS``) to keep memory use down.
    """
    df_travel = pd.read_parquet(travel_times_parquet, engine="pyarrow", columns=TRAVEL_COLUMNS)
    df_dwell = pd.read_parquet(dwell_times_parquet, engine="pyarrow", columns=DWELL_COLUMNS)
    for df in (df_travel, df_dwell):
        if df["from_time"].dt.tz is None:
            df["from_time"] = df["from_time"].dt.tz_localize("UTC")
    logger.info(
        "Loaded %d travel time and %d dwell time observations",
        len(df_travel), len(df_dwell),
    )
    return df_travel, df_dwell


def _file_municipalities(parquet_path: str | Path) -> set[str]:
    """Municipalities in one observation file, reading only that column."""
    table = pq.read_table(parquet_path, columns=["municipality"], read_dictionary=["municipality"])
    municipalities = set(table["municipality"].unique().to_pylist())
    del table
    pa.default_memory_pool().release_unused()
    return municipalities


def observation_municipalities(
    travel_times_parquet: str | Path,
    dwell_times_parquet: str | Path,
) -> set[str]:
    """Municipalities present in both files, without loading the observations."""
    return _file_municipalities(travel_times_parquet) & _file_municipalities(dwell_times_parquet)


def _read_municipalities(
    parquet_path: str | Path,
    columns: List[str],
    municipalities: List[str],
) -> pd.DataFrame:
    """Read the rows of some municipalities from disk, in file order (UTC timestamps).

    Matches ``df[df["municipality"].isin(municipalities)]`` on the full table.
    Rows come back in file order (checked against row-group reads on the full data).
    """
    if municipalities:
        df = pd.read_parquet(
            parquet_path,
            engine="pyarrow",
            columns=columns,
            filters=[("municipality", "in", municipalities)],
            use_threads=False,
        )
    else:
        # pyarrow rejects an empty "in" filter
        df = pq.read_schema(parquet_path).empty_table().select(columns).to_pandas()
    pa.default_memory_pool().release_unused()
    if df["from_time"].dt.tz is None:
        df["from_time"] = df["from_time"].dt.tz_localize("UTC")
    return df


def load_observation_groups(
    travel_times_parquet: str | Path,
    dwell_times_parquet: str | Path,
    regions: Dict[str, List[str]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Travel and dwell observations per region group, read directly from disk.

    Each group holds the same rows in the same order as filtering the output
    of ``load_observations`` with ``isin``, but the full tables are never in
    memory. Index labels may differ.
    """
    travel = {
        name: _read_municipalities(travel_times_parquet, TRAVEL_COLUMNS, municipalities)
        for name, municipalities in regions.items()
    }
    dwell = {
        name: _read_municipalities(dwell_times_parquet, DWELL_COLUMNS, municipalities)
        for name, municipalities in regions.items()
    }
    return travel, dwell


class GlobalScaler:
    """Z-score scaler with a single global mean and standard deviation."""

    def __init__(self, eps: float = 1e-8):
        self.mean_: np.ndarray = np.array([0.0])
        self.scale_: np.ndarray = np.array([1.0])
        self.eps = eps
        self._fitted = False

    @property
    def mean(self) -> float:
        return float(self.mean_[0])

    @property
    def std(self) -> float:
        return float(self.scale_[0])

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, values: np.ndarray) -> "GlobalScaler":
        """Fit on all non-NaN values."""
        values = values.flatten()
        values = values[~np.isnan(values)]
        if len(values) == 0:
            logger.warning("GlobalScaler.fit() received no valid values, using defaults")
            self.mean_ = np.array([0.0])
            self.scale_ = np.array([1.0])
        else:
            self.mean_ = np.array([float(np.mean(values))])
            std = float(np.std(values))
            self.scale_ = np.array([std if std > self.eps else self.eps])
        self._fitted = True
        return self

    def inverse_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean


def compute_time_context(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Time context of each timestamp, ``[n, TIME_CONTEXT_DIM]``.

    Filled in chunks, so the temporary arrays stay small.
    """
    n = len(timestamps)
    time_context = np.empty((n, TIME_CONTEXT_DIM), dtype=np.float32)
    for start in range(0, n, TIME_CONTEXT_CHUNK_ROWS):
        stop = start + TIME_CONTEXT_CHUNK_ROWS
        time_context[start:stop] = _time_context_chunk(timestamps[start:stop])
    return time_context


def _time_context_chunk(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Time context of one chunk of timestamps."""
    minutes = timestamps.hour * 60 + timestamps.minute
    tod_fraction = minutes / 1440.0
    sin_tod = np.sin(2 * np.pi * tod_fraction)
    cos_tod = np.cos(2 * np.pi * tod_fraction)

    dow = timestamps.dayofweek.values  # 0=Monday, 6=Sunday
    dow_onehot = np.zeros((len(timestamps), 7), dtype=np.float32)
    dow_onehot[np.arange(len(timestamps)), dow] = 1.0

    month_fraction = (timestamps.month - 1) / 12.0
    sin_month = np.sin(2 * np.pi * month_fraction)
    cos_month = np.cos(2 * np.pi * month_fraction)

    # Look up holidays once per day
    day_keys = timestamps.normalize()
    unique_days = day_keys.unique()
    holiday_lookup = pd.Series(
        [1.0 if d.date() in _NL_HOLIDAYS else 0.0 for d in unique_days],
        index=unique_days,
        dtype=np.float32,
    )
    is_holiday = day_keys.map(holiday_lookup).to_numpy(dtype=np.float32)

    time_context = np.column_stack(
        [sin_tod, cos_tod, dow_onehot, sin_month, cos_month, is_holiday]
    )
    return time_context.astype(np.float32)


def _release_columns(df: pd.DataFrame, columns: Tuple[str, ...]) -> None:
    """Delete columns in place if present (``del`` does not warn on filtered frames)."""
    for column in columns:
        if column in df.columns:
            del df[column]


class ProxyTaskBatchIterator:
    """Batch iterator over the link and stop samples of a dataset.

    Yields :data:`Batch` tuples by slicing the dataset tensors. With
    ``subsample < 1.0`` each pass draws a fresh random subset of that
    fraction of the samples (requires ``shuffle=True``).
    """

    def __init__(
        self,
        dataset: "RawObservationProxyDataset",
        batch_size: int,
        shuffle: bool = False,
        pin_memory: bool = False,
        subsample: float = 1.0,
    ) -> None:
        if not (0.0 < subsample <= 1.0):
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")
        if subsample < 1.0 and not shuffle:
            raise ValueError("subsample < 1.0 requires shuffle=True")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self._n_link = dataset.num_link_samples
        self._n_stop = dataset.num_stop_samples
        self._n_total = self._n_link + self._n_stop
        self._n_used = int(self._n_total * subsample)

    def __len__(self) -> int:
        return (self._n_used + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        if self.shuffle:
            perm = torch.randperm(self._n_total)[: self._n_used]
        else:
            perm = torch.arange(self._n_total)

        ds = self.dataset
        ctx_dim = ds._link_time_context.shape[1]

        for start in range(0, self._n_used, self.batch_size):
            batch_idx = perm[start : start + self.batch_size]

            link_mask = batch_idx < self._n_link
            link_sel = batch_idx[link_mask]
            stop_sel = batch_idx[~link_mask] - self._n_link

            if link_sel.numel() > 0:
                link_indices = ds._link_node_indices[link_sel]
                link_ctx = ds._link_time_context[link_sel]
                link_targets = ds._link_targets[link_sel]
            else:
                link_indices = torch.empty(0, dtype=torch.long)
                link_ctx = torch.empty(0, ctx_dim, dtype=torch.float32)
                link_targets = torch.empty(0, dtype=torch.float32)

            stop_labels: Optional[torch.Tensor]
            if stop_sel.numel() > 0:
                stop_indices = ds._stop_node_indices[stop_sel]
                stop_ctx = ds._stop_time_context[stop_sel]
                stop_targets = ds._stop_targets[stop_sel]
                stop_labels = ds._stop_labels[stop_sel]
            else:
                stop_indices = torch.empty(0, dtype=torch.long)
                stop_ctx = torch.empty(0, ctx_dim, dtype=torch.float32)
                stop_targets = torch.empty(0, dtype=torch.float32)
                stop_labels = None

            if self.pin_memory:
                link_indices = link_indices.pin_memory()
                link_ctx = link_ctx.pin_memory()
                link_targets = link_targets.pin_memory()
                stop_indices = stop_indices.pin_memory()
                stop_ctx = stop_ctx.pin_memory()
                stop_targets = stop_targets.pin_memory()
                if stop_labels is not None:
                    stop_labels = stop_labels.pin_memory()

            yield (
                link_indices, link_ctx, link_targets,
                stop_indices, stop_ctx, stop_targets,
                stop_labels,
            )


class RawObservationProxyDataset:
    """Proxy-task samples built from individual travel and dwell time observations.

    Args:
        df_travel: Travel time observations (``link``, ``from_time``, ``travel_time``).
        df_dwell: Dwell time observations (``stop``, ``from_time``, ``travel_time``).
        link_nodes: Link nodes of the graph; their row order defines the node indices.
        stop_nodes: Stop nodes of the graph; their row order defines the node indices.
        freeflow_times: Ideal travel time per link ID (seconds).
        freeflow_ratio: Link target is ``travel_time / ideal`` (True) or
            ``travel_time - ideal`` (False).
        log_transform: Model travel and dwell times in log1p space (difference
            target only).
        link_scaler, stop_scaler: Fitted target scalers; fitted on this
            dataset if ``None`` (the dwell scaler on stopped observations only).
        release_inputs: Delete columns from ``df_travel`` and ``df_dwell`` in
            place once they are used, to lower peak memory. The frames are
            not usable afterwards. If False, they are left unmodified.
    """

    def __init__(
        self,
        df_travel: pd.DataFrame,
        df_dwell: pd.DataFrame,
        link_nodes: pd.DataFrame,
        stop_nodes: pd.DataFrame,
        freeflow_times: Dict[str, float],
        freeflow_ratio: bool,
        log_transform: bool = False,
        link_scaler: Optional[GlobalScaler] = None,
        stop_scaler: Optional[GlobalScaler] = None,
        release_inputs: bool = False,
    ):
        if freeflow_ratio and log_transform:
            raise ValueError("The travel time ratio target cannot be combined with log_transform")

        missing_travel = {"link", "from_time", "travel_time"} - set(df_travel.columns)
        if missing_travel:
            raise ValueError(f"df_travel missing required columns: {missing_travel}")
        missing_dwell = {"stop", "from_time", "travel_time"} - set(df_dwell.columns)
        if missing_dwell:
            raise ValueError(f"df_dwell missing required columns: {missing_dwell}")

        self.link_scaler = link_scaler if link_scaler is not None else GlobalScaler()
        self.stop_scaler = stop_scaler if stop_scaler is not None else GlobalScaler()

        link_id_to_idx = {str(node_id): idx for idx, node_id in enumerate(link_nodes["id"])}
        stop_id_to_idx = {str(node_id): idx for idx, node_id in enumerate(stop_nodes["id"])}

        # The input frames are not copied, and only modified with release_inputs:
        # only the key Series and the needed columns of the matching rows are
        # materialized.

        # --- Link (travel time) samples ---
        link_keys = df_travel["link"].astype(str)
        link_valid = link_keys.isin(link_id_to_idx).to_numpy()
        link_keys = link_keys[link_valid]
        num_link_valid = len(link_keys)
        if num_link_valid < len(df_travel):
            logger.debug("Dropped %d travel times of links not in the graph",
                         len(df_travel) - num_link_valid)
        if num_link_valid == 0:
            raise ValueError("No travel time observations match the link nodes of the graph")

        link_travel_times = df_travel.loc[link_valid, "travel_time"].to_numpy()
        ff_values = link_keys.map(freeflow_times).to_numpy(dtype=np.float64)
        missing_ff = np.isnan(ff_values)
        if missing_ff.any():
            missing_ids = link_keys[missing_ff].unique()
            raise ValueError(
                f"Missing ideal travel times for {missing_ff.sum()} observations "
                f"({len(missing_ids)} links), e.g. {list(missing_ids[:5])}"
            )
        del missing_ff
        # Per-observation reference used to convert predictions back to seconds
        if freeflow_ratio:
            ff_safe = np.maximum(ff_values, 1.0)  # avoid division by zero
            link_values = link_travel_times / ff_safe
            self._link_freeflow_refs = torch.from_numpy(ff_safe.astype(np.float32))
            del ff_safe
        elif log_transform:
            link_values = np.log1p(link_travel_times) - np.log1p(ff_values)
            self._link_freeflow_refs = torch.from_numpy(np.log1p(ff_values).astype(np.float32))
        else:
            link_values = link_travel_times - ff_values
            self._link_freeflow_refs = torch.from_numpy(ff_values.astype(np.float32))
        del link_travel_times, ff_values

        if not self.link_scaler.is_fitted:
            self.link_scaler.fit(link_values)

        link_node_indices = link_keys.map(link_id_to_idx).values
        del link_keys
        self._link_node_indices = torch.from_numpy(link_node_indices.astype(np.int64))
        del link_node_indices
        self._link_municipalities: Optional[np.ndarray] = (
            df_travel.loc[link_valid, "municipality"].values
            if "municipality" in df_travel.columns else None
        )
        normalized_link_times = (link_values - self.link_scaler.mean) / self.link_scaler.std
        del link_values
        self._link_targets = torch.from_numpy(normalized_link_times.astype(np.float32))
        del normalized_link_times
        if release_inputs:
            _release_columns(df_travel, ("link", "travel_time", "municipality"))
        self._link_time_context = torch.from_numpy(
            compute_time_context(pd.DatetimeIndex(df_travel.loc[link_valid, "from_time"]))
        )
        del link_valid
        if release_inputs:
            _release_columns(df_travel, ("from_time",))
        self._num_link_samples = num_link_valid

        # --- Stop (dwell time) samples ---
        # Dwell rows hold the stop ID without the "stop_" prefix of the node IDs.
        # Matching on the bare ID avoids building one prefixed string per row.
        prefix = "stop_"
        bare_stop_id_to_idx = {
            node_id[len(prefix):]: idx
            for node_id, idx in stop_id_to_idx.items()
            if node_id.startswith(prefix)
        }
        stop_keys = df_dwell["stop"].astype(str)
        stop_valid = stop_keys.isin(bare_stop_id_to_idx).to_numpy()
        stop_keys = stop_keys[stop_valid]
        num_stop_valid = len(stop_keys)
        if num_stop_valid < len(df_dwell):
            logger.debug("Dropped %d dwell times of stops not in the graph",
                         len(df_dwell) - num_stop_valid)
        if num_stop_valid == 0:
            raise ValueError("No dwell time observations match the stop nodes of the graph")

        dwell_times = df_dwell.loc[stop_valid, "travel_time"].to_numpy()  # dwell time in seconds
        stopped_mask = dwell_times > 0
        if log_transform:
            dwell_times = np.log1p(dwell_times)

        if not self.stop_scaler.is_fitted:
            stopped_dwell_values = dwell_times[stopped_mask]
            if len(stopped_dwell_values) == 0:
                raise ValueError("No stopped observations (dwell > 0) to fit the dwell time scaler")
            self.stop_scaler.fit(stopped_dwell_values)
            del stopped_dwell_values

        stop_node_indices = stop_keys.map(bare_stop_id_to_idx).values
        del stop_keys
        self._stop_node_indices = torch.from_numpy(stop_node_indices.astype(np.int64))
        del stop_node_indices
        normalized_dwell_times = (dwell_times - self.stop_scaler.mean) / self.stop_scaler.std
        del dwell_times
        self._stop_targets = torch.from_numpy(normalized_dwell_times.astype(np.float32))
        del normalized_dwell_times
        self._stop_labels = torch.from_numpy(stopped_mask.astype(np.float32))
        # Aligned with the stopped observations (the dwell time regression targets)
        self._stop_municipalities: Optional[np.ndarray] = (
            df_dwell.loc[stop_valid, "municipality"].values[stopped_mask]
            if "municipality" in df_dwell.columns else None
        )
        del stopped_mask
        if release_inputs:
            _release_columns(df_dwell, ("stop", "travel_time", "municipality"))
        self._stop_time_context = torch.from_numpy(
            compute_time_context(pd.DatetimeIndex(df_dwell.loc[stop_valid, "from_time"]))
        )
        self._num_stop_samples = num_stop_valid

        logger.info("Created %d link samples and %d stop samples",
                    self._num_link_samples, self._num_stop_samples)

    def __len__(self) -> int:
        return self._num_link_samples + self._num_stop_samples

    @property
    def num_link_samples(self) -> int:
        return self._num_link_samples

    @property
    def num_stop_samples(self) -> int:
        return self._num_stop_samples

    @property
    def link_municipalities(self) -> Optional[np.ndarray]:
        """Municipality of each link observation."""
        return self._link_municipalities

    @property
    def stop_municipalities(self) -> Optional[np.ndarray]:
        """Municipality of each stopped dwell time observation."""
        return self._stop_municipalities

    @property
    def link_freeflow_refs(self) -> torch.Tensor:
        """Per-observation ideal travel time used to convert link predictions to seconds."""
        return self._link_freeflow_refs
