"""Memory test of the training data path, without training a model.

Runs the data steps of train_embeddings.main() for the last (biggest) fold:
load, region groups, graphs, fold boundaries, datasets, observation frames and
the seasonal naive scoring with dummy predictions. Test subsets are built but
never scored, and no score is printed or saved.

Usage: python -I memtest.py TRAVEL_PARQUET DWELL_PARQUET ARTIFACTS_DIR LOG_FILE [REGIONS_JSON]
REGIONS_JSON (small tests only) sets {"train": [...], "val": [...], "test": [...]} by hand.
"""
import json
import logging
import os
import sys
import threading
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa

from bus_embeddings.cli.train_embeddings import SUBSETS, FoldData
from bus_embeddings.data.proxy_task_dataset import load_observation_groups, observation_municipalities
from bus_embeddings.training.config import MODEL_DEFAULTS
from bus_embeddings.training.metrics import (
    compute_seasonal_naive_metrics,
    compute_seasonal_naive_metrics_by_region,
)
from bus_embeddings.training.train_utils import uses_ratio_target
from bus_embeddings.utils.hetero_graph_utils import build_hetero_graph
from bus_embeddings.utils.splits import (
    assign_regions_to_splits,
    discover_regions,
    fold_boundaries,
    region_artifact_paths,
)

TRAVEL, DWELL, ARTIFACTS, LOG = sys.argv[1:5]
REGIONS_OVERRIDE = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
MIN_AVAILABLE_MB = 600
# Dummy zero predictions make log(0) in the geometric mean; the results are discarded anyway
warnings.filterwarnings("ignore", category=RuntimeWarning)

log = open(LOG, "w")
logging.basicConfig(stream=log, level=logging.INFO, format="%(levelname)-8s | %(name)s: %(message)s")
logging.getLogger("bus_embeddings.training.metrics").setLevel(logging.WARNING)


def say(msg: str) -> None:
    log.write(msg + "\n")
    log.flush()


def status_kb(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


def available_mb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


# Watchdog: stops the run before the PC runs out of memory, and samples RSS
# so each step's peak is known even if clear_refs isn't allowed.
current_step = "start"
step_peak_kb = 0


def watchdog() -> None:
    global step_peak_kb
    while True:
        step_peak_kb = max(step_peak_kb, status_kb("VmRSS"))
        if available_mb() < MIN_AVAILABLE_MB:
            say(f"WATCHDOG: MemAvailable below {MIN_AVAILABLE_MB} MB during step '{current_step}', "
                f"RSS {status_kb('VmRSS') / 1048576:.2f} GB. Stopping.")
            os._exit(3)
        time.sleep(0.2)


threading.Thread(target=watchdog, daemon=True).start()
rows = []


def step(name, fn):
    """Run fn, record wall time, RSS after and peak during."""
    global current_step, step_peak_kb
    current_step = name
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass
    step_peak_kb = status_kb("VmRSS")
    t0 = time.time()
    out = fn()
    wall = time.time() - t0
    peak = max(status_kb("VmHWM"), step_peak_kb)
    rss = status_kb("VmRSS")
    rows.append((name, wall, rss, peak))
    say(f"STEP {name}: {wall:.1f} s, RSS after {rss / 1048576:.2f} GB, peak during {peak / 1048576:.2f} GB")
    return out


say(f"Start: MemAvailable {available_mb()} MB")
hyperparams = dict(MODEL_DEFAULTS["gat"])
ratio_target = uses_ratio_target(hyperparams)

# a. Municipalities and region groups, read directly from disk (main(), patched)
if REGIONS_OVERRIDE is None:
    municipalities = step("a municipalities", lambda: observation_municipalities(TRAVEL, DWELL))
    train_regions, val_regions, test_regions = assign_regions_to_splits(discover_regions(ARTIFACTS, municipalities))
    regions = {"train": train_regions, "val": val_regions, "test": test_regions}
else:
    regions = REGIONS_OVERRIDE

# Test group: only its time range is kept until test_spatial (main(), patched)
test_group = {"test": regions["test"]}


def test_extremes_fn():
    t, d = load_observation_groups(TRAVEL, DWELL, test_group)
    return [e for df in (t["test"], d["test"]) for e in (df["from_time"].min(), df["from_time"].max())]


test_time_extremes = step("b0 test group time range", test_extremes_fn)
travel, dwell = step("b load train and val groups", lambda: load_observation_groups(
    TRAVEL, DWELL, {"train": regions["train"], "val": regions["val"]}))
say(f"Arrow pool after load: {pa.default_memory_pool().bytes_allocated() / 1048576:.0f} MB allocated")


# c. Graphs (main())
def build_graphs():
    graphs, link_nodes, stop_nodes = {}, {}, {}
    norm_stats = None
    for group in ("train", "val", "test"):
        graphs[group], link_nodes[group], stop_nodes[group], stats = build_hetero_graph(
            *region_artifact_paths(regions[group], ARTIFACTS),
            feature_config=hyperparams,
            feature_norm_stats=norm_stats,
        )
        if norm_stats is None:
            norm_stats = stats
    freeflow_times = {
        group: dict(zip(nodes["id"].astype(str), nodes["osm_travel_time"].astype(float)))
        for group, nodes in link_nodes.items()
    }
    return graphs, link_nodes, stop_nodes, freeflow_times


graphs, link_nodes, stop_nodes, freeflow_times = step("c graphs", build_graphs)


# d. Fold boundaries (main(), patched)
def boundaries_fn():
    time_range = pd.DataFrame({"from_time": pd.Series([
        extreme
        for frames in (travel, dwell)
        for df in frames.values()
        for extreme in (df["from_time"].min(), df["from_time"].max())
    ] + test_time_extremes, dtype=travel["train"]["from_time"].dtype)})
    return fold_boundaries(time_range, time_range)


boundaries = step("d fold boundaries", boundaries_fn)
for i, (s, e, _, _) in enumerate(boundaries):
    n_tt = int(((travel["train"]["from_time"] >= s) & (travel["train"]["from_time"] < e)).sum())
    n_dw = int(((dwell["train"]["from_time"] >= s) & (dwell["train"]["from_time"] < e)).sum())
    say(f"fold {i}: train window travel {n_tt:,} rows, dwell {n_dw:,} rows")

# e. Last fold: datasets as during training
train_start, train_end, val_end, test_end = boundaries[-1]
data = FoldData(
    travel, dwell, link_nodes, stop_nodes, freeflow_times,
    periods={"train": (train_start, train_end), "val": (train_end, val_end), "test": (val_end, test_end)},
    ratio_target=ratio_target,
    log_transform=hyperparams["log-transform"],
)
train_dataset = step("e train dataset", lambda: data.dataset("train"))
val_dataset = step("e val_spatial dataset", lambda: data.dataset("val_spatial", fitted=train_dataset))
val_observations = step("e val_spatial observations (held during training)",
                        lambda: data.observations("val_spatial"))


def naive_scoring(dataset, observations, per_region):
    """What add_scaled_errors does, with dummy predictions; results are discarded."""
    for node_type, n, regions_arr in (
        ("link", dataset.num_link_samples, dataset.link_municipalities),
        ("stop", len(dataset.stop_municipalities), dataset.stop_municipalities),
    ):
        preds = np.zeros(n, dtype=np.float32)
        targets = np.zeros(n, dtype=np.float32)
        df_eval, df_history = observations[node_type]
        if per_region:
            compute_seasonal_naive_metrics_by_region(df_eval, df_history, node_type, preds, targets, regions_arr)
        else:
            compute_seasonal_naive_metrics(df_eval, df_history, node_type, model_mae=0.0, model_rmse=0.0)


# Scoring done every epoch on val_spatial during training
step("e val_spatial naive scoring (every epoch)", lambda: naive_scoring(val_dataset, val_observations, True))
del val_observations

# f. Evaluation loop of main(): train and val_temporal are scored, test subsets only built
for subset in ("train", "val_temporal", "test_temporal", "test_spatial"):
    assert subset in SUBSETS
    if subset == "test_spatial":
        t_new, d_new = step("f load test group on demand", lambda: load_observation_groups(TRAVEL, DWELL, test_group))
        travel.update(t_new)
        dwell.update(d_new)
        del t_new, d_new
    dataset = train_dataset if subset == "train" else step(
        f"f {subset} dataset", lambda: data.dataset(subset, fitted=train_dataset))
    observations = step(f"f {subset} observations", lambda: data.observations(subset))
    if subset in ("train", "val_temporal"):
        step(f"f {subset} naive scoring", lambda: naive_scoring(dataset, observations, subset != "train"))
    del observations
    if subset != "train":
        del dataset
    if subset == "test_spatial":
        del travel["test"], dwell["test"]

say("")
say(f"{'step':<55} {'wall s':>7} {'RSS after GB':>13} {'peak GB':>8}")
for name, wall, rss, peak in rows:
    say(f"{name:<55} {wall:>7.1f} {rss / 1048576:>13.2f} {peak / 1048576:>8.2f}")
overall = max(r[3] for r in rows)
worst = max(rows, key=lambda r: r[3])[0]
say(f"OVERALL PEAK {overall / 1048576:.2f} GB at step '{worst}'")
say("DONE")
