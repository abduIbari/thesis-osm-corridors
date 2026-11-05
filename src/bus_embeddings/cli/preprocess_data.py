"""Preprocess raw CSV data into cleaned Parquet files for fast training."""
import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bus_embeddings.logging import get_logger, setup_logging
from bus_embeddings.data.loaders import iter_travel_times, iter_dwell_times
from bus_embeddings.data.thresholds import (
    filter_travel_time_freeflow_ratio,
    filter_dwell_times,
)

logger = get_logger(__name__)

# Column types of the output files (same as the former single df.to_parquet)
TRAVEL_SCHEMA = pa.schema([
    ("from_stop", pa.string()),
    ("to_stop", pa.string()),
    ("from_time", pa.timestamp("ns", tz="UTC")),
    ("to_time", pa.timestamp("ns", tz="UTC")),
    ("travel_time", pa.float64()),
    ("link", pa.string()),
    ("municipality", pa.string()),
])
DWELL_SCHEMA = pa.schema([
    ("stop", pa.string()),
    ("from_time", pa.timestamp("ns", tz="UTC")),
    ("to_time", pa.timestamp("ns", tz="UTC")),
    ("travel_time", pa.float64()),
    ("municipality", pa.string()),
])


class _ParquetAppender:
    """Append DataFrame pieces to one parquet file with a fixed schema.

    Writes to ``<name>.tmp`` and renames it to the final name only when the
    block exits without error, so an aborted run leaves no complete-looking file.
    """

    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path = path
        self.tmp_path = path.with_name(path.name + ".tmp")
        self.schema = schema
        self.rows = 0
        self._writer: pq.ParquetWriter | None = None

    def write(self, df: pd.DataFrame) -> None:
        # Cast to the fixed schema, so empty pieces can't change column types
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        if self._writer is None:
            # The first piece's schema also carries the pandas dtype metadata
            self._writer = pq.ParquetWriter(self.tmp_path, table.schema)
        if table.num_rows > 0:
            self._writer.write_table(table)
            self.rows += table.num_rows

    def __enter__(self) -> "_ParquetAppender":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            try:
                if self._writer is not None:
                    self._writer.close()
            finally:
                self.tmp_path.unlink(missing_ok=True)
            return
        if self._writer is None:  # no pieces at all: still write an empty file
            self._writer = pq.ParquetWriter(self.tmp_path, self.schema)
        self._writer.close()
        self.tmp_path.replace(self.path)


def _discover_csv_regions(
    travel_times_dir: str,
    dwell_times_dir: str,
) -> list[str]:
    """Return sorted municipality IDs that have both travel and dwell CSVs."""
    travel_path = Path(travel_times_dir)
    dwell_path = Path(dwell_times_dir)

    if not travel_path.is_dir():
        raise ValueError(f"Travel times directory does not exist: {travel_times_dir}")
    if not dwell_path.is_dir():
        raise ValueError(f"Dwell times directory does not exist: {dwell_times_dir}")

    travel_regions = {p.stem for p in travel_path.glob("*.csv")}
    dwell_regions = {p.stem for p in dwell_path.glob("*.csv")}
    available = sorted(travel_regions & dwell_regions)

    if not available:
        raise ValueError(
            f"No municipalities found with both travel and dwell time CSVs. "
            f"Travel: {sorted(travel_regions)}, Dwell: {sorted(dwell_regions)}"
        )
    return available


def _load_link_nodes(artifacts_dir: Path, region: str) -> pd.DataFrame:
    path = artifacts_dir / region / "nodes_link.csv"
    if not path.exists():
        raise ValueError(f"Missing link node artifacts for region {region}: {path}")
    return pd.read_csv(path, usecols=["id", "osm_travel_time"])


def _load_stop_nodes(artifacts_dir: Path, region: str) -> pd.DataFrame:
    path = artifacts_dir / region / "nodes_stop.csv"
    if not path.exists():
        raise ValueError(f"Missing stop node artifacts for region {region}: {path}")
    return pd.read_csv(path, usecols=["stop_id", "gtfs_is_terminus"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw CSV travel/dwell data into cleaned Parquet files."
    )
    parser.add_argument(
        "--travel-times-dir",
        type=str,
        required=True,
        help="Directory containing {municipality_id}.csv travel time files.",
    )
    parser.add_argument(
        "--dwell-times-dir",
        type=str,
        required=True,
        help="Directory containing {municipality_id}.csv dwell time files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        required=True,
        help="Directory containing per-municipality graph artifacts "
             "(each with nodes_link.csv and nodes_stop.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for parquet files.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=2_000_000,
        help="Rows read from a CSV per piece (bounds peak memory).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    regions = _discover_csv_regions(args.travel_times_dir, args.dwell_times_dir)
    logger.info("Discovered %d municipalities: %s", len(regions), regions)

    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fixed-bound filters applied to all regions regardless of split
    # (see data/thresholds.py):
    #   - Travel: keep travel_time / osm_travel_time within the freeflow-ratio
    #     bounds; rows without a link node (e.g. self-loops) are dropped too
    #   - Dwell:  drop terminus stops and dwells above DWELL_TIME_CAP
    # ------------------------------------------------------------------
    # Each CSV is read in pieces, and each cleaned piece is appended to the
    # output at once, so memory holds one piece at a time.
    travel_out = output_dir / "travel_times.parquet"
    dwell_out = output_dir / "dwell_times.parquet"

    with (
        _ParquetAppender(travel_out, TRAVEL_SCHEMA) as travel_writer,
        _ParquetAppender(dwell_out, DWELL_SCHEMA) as dwell_writer,
    ):
        for region in regions:
            tt_path = str(Path(args.travel_times_dir) / f"{region}.csv")
            dw_path = str(Path(args.dwell_times_dir) / f"{region}.csv")

            link_nodes = _load_link_nodes(artifacts_dir, region)
            stop_nodes = _load_stop_nodes(artifacts_dir, region)

            n_tt_before = n_tt_after = 0
            for df_tt in iter_travel_times(tt_path, chunk_rows=args.chunk_rows):
                n_tt_before += len(df_tt)
                df_tt = filter_travel_time_freeflow_ratio(df_tt, link_nodes)
                df_tt["municipality"] = region
                travel_writer.write(df_tt)
                n_tt_after += len(df_tt)

            n_dw_before = n_dw_after = 0
            for df_dw in iter_dwell_times(dw_path, chunk_rows=args.chunk_rows):
                n_dw_before += len(df_dw)
                df_dw = filter_dwell_times(df_dw, stop_nodes)
                df_dw["municipality"] = region
                dwell_writer.write(df_dw)
                n_dw_after += len(df_dw)

            logger.info(
                "Region %s: travel %d → %d (−%.1f%%), dwell %d → %d (−%.1f%%)",
                region,
                n_tt_before, n_tt_after,
                100.0 * (n_tt_before - n_tt_after) / max(n_tt_before, 1),
                n_dw_before, n_dw_after,
                100.0 * (n_dw_before - n_dw_after) / max(n_dw_before, 1),
            )

    logger.info(
        "Combined: %d travel time records, %d dwell time records",
        travel_writer.rows, dwell_writer.rows,
    )
    logger.info("Saved travel times → %s (%d rows)", travel_out, travel_writer.rows)
    logger.info("Saved dwell times  → %s (%d rows)", dwell_out, dwell_writer.rows)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
