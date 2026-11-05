"""Data loading utilities for travel time and dwell time datasets."""
from pathlib import Path
from typing import Iterator, List, Union

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from pandas._libs.parsers import STR_NA_VALUES

from bus_embeddings.logging import get_logger

logger = get_logger(__name__)

_TRAVEL_COLUMNS = ["from_stop", "to_stop", "from_time", "to_time"]
_DWELL_COLUMNS = ["stop", "from_time", "to_time"]
# What pyarrow infers for "2022-01-01 09:18:07+0000"; pd.to_datetime turns it into
# datetime64[ns, UTC], matching load_*_times. Fractional seconds would fail loudly.
_TIME_TYPE = pa.timestamp("s", tz="UTC")


def _prepare_travel(df: pd.DataFrame) -> pd.DataFrame:
    """Cast ids to str, parse times, add ``travel_time`` and ``link``."""
    df = df.astype({"from_stop": str, "to_stop": str})
    df["from_time"] = pd.to_datetime(df["from_time"], utc=True)
    df["to_time"] = pd.to_datetime(df["to_time"], utc=True)
    df["travel_time"] = (df["to_time"] - df["from_time"]).dt.total_seconds()
    df["link"] = "link_" + df["from_stop"] + "_" + df["to_stop"]
    return df


def _prepare_dwell(df: pd.DataFrame) -> pd.DataFrame:
    """Cast stop to str, parse times, add ``travel_time`` (the dwell duration)."""
    df = df.astype({"stop": str})
    df["from_time"] = pd.to_datetime(df["from_time"], utc=True)
    df["to_time"] = pd.to_datetime(df["to_time"], utc=True)
    df["travel_time"] = (df["to_time"] - df["from_time"]).dt.total_seconds()
    return df


def load_travel_times(path: Union[str, Path]) -> pd.DataFrame:
    """Load travel times from CSV, computing duration and link IDs."""
    df = pd.read_csv(
        path,
        usecols=_TRAVEL_COLUMNS,
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    df = _prepare_travel(df)

    logger.debug("Loaded %d travel time records from '%s'", len(df), path)
    return df


def load_dwell_times(path: Union[str, Path]) -> pd.DataFrame:
    """Load dwell times from CSV, computing duration (in the ``travel_time`` column)."""
    df = pd.read_csv(
        path,
        usecols=_DWELL_COLUMNS,
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    df = _prepare_dwell(df)

    logger.debug("Loaded %d dwell time records from '%s'", len(df), path)
    return df


def _iter_csv_chunks(
    path: Union[str, Path],
    columns: List[str],
    column_types: dict,
    chunk_rows: int,
) -> Iterator[pd.DataFrame]:
    """Stream ``columns`` of a CSV as pyarrow-backed DataFrames of about ``chunk_rows`` rows.

    Column types are pinned so every chunk gets the same dtypes, whatever the
    first block of the file looks like. Null values follow pandas' defaults,
    as in ``pd.read_csv(engine="pyarrow")``.
    """
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")

    reader = pacsv.open_csv(
        path,
        convert_options=pacsv.ConvertOptions(
            include_columns=columns,
            column_types=column_types,
            null_values=sorted(STR_NA_VALUES),
            strings_can_be_null=True,
        ),
    )
    # Gather reader blocks until a chunk has at least chunk_rows rows
    pending: List[pa.RecordBatch] = []
    n_pending = 0
    for batch in reader:
        pending.append(batch)
        n_pending += batch.num_rows
        if n_pending >= chunk_rows:
            yield pa.Table.from_batches(pending, schema=reader.schema).to_pandas(types_mapper=pd.ArrowDtype)
            pending, n_pending = [], 0
    if n_pending:
        yield pa.Table.from_batches(pending, schema=reader.schema).to_pandas(types_mapper=pd.ArrowDtype)


def iter_travel_times(
    path: Union[str, Path], chunk_rows: int = 2_000_000
) -> Iterator[pd.DataFrame]:
    """Yield travel times in chunks of about ``chunk_rows`` rows.

    Together the chunks equal :func:`load_travel_times`, with two exceptions:
    ids are always read as text (an all-digit id keeps leading zeros), and a
    time with fractional seconds raises an error.
    """
    column_types = {
        "from_stop": pa.string(),
        "to_stop": pa.string(),
        "from_time": _TIME_TYPE,
        "to_time": _TIME_TYPE,
    }
    n = 0
    for chunk in _iter_csv_chunks(path, _TRAVEL_COLUMNS, column_types, chunk_rows):
        n += len(chunk)
        yield _prepare_travel(chunk)
    logger.debug("Streamed %d travel time records from '%s'", n, path)


def iter_dwell_times(
    path: Union[str, Path], chunk_rows: int = 2_000_000
) -> Iterator[pd.DataFrame]:
    """Yield dwell times in chunks of about ``chunk_rows`` rows; see :func:`iter_travel_times`."""
    column_types = {
        "stop": pa.string(),
        "from_time": _TIME_TYPE,
        "to_time": _TIME_TYPE,
    }
    n = 0
    for chunk in _iter_csv_chunks(path, _DWELL_COLUMNS, column_types, chunk_rows):
        n += len(chunk)
        yield _prepare_dwell(chunk)
    logger.debug("Streamed %d dwell time records from '%s'", n, path)
