"""Per-city link scores of a finished training run, from its saved fold models (no training).

Horn's code scores every val and test city on its own, combines the cities with the
geometric mean, and saves only that mean in metrics.json. This script loads each fold's
saved model (attempt_<N>/fold_<k>.pt), scores the val-spatial and test-spatial subsets
again with Horn's own functions, and keeps each city's link MASE and RMSSE.

Check: per fold, the geometric mean over cities must give back the link MASE and RMSSE
in metrics.json; the script stops with an error if they differ.

Run from ~/thesis with Horn's install, one run at a time and not during a training run
(it loads the val and test observations, like training does):

    bus-embeddings/.venv/bin/python bus-embeddings/thesis/city_scores.py data/runs/L0/run1_seed42

Writes <run>/city_scores.csv: subset, fold, city, observations, mase, rmsse.
"""

import argparse
import csv
import json
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bus_embeddings.cli.train_embeddings import SUBSETS, FoldData, evaluate
from bus_embeddings.data.proxy_task_dataset import (
    _read_municipalities,
    load_observation_groups,
    observation_municipalities,
)
from bus_embeddings.models.checkpoint import load_checkpoint
from bus_embeddings.training.config import FEATURE_CONFIG
from bus_embeddings.training.metrics import _scale_by_naive, seasonal_naive_errors_by_region
from bus_embeddings.training.train_utils import uses_ratio_target
from bus_embeddings.utils.hetero_graph_utils import build_hetero_graph
from bus_embeddings.utils.splits import (
    assign_regions_to_splits,
    discover_regions,
    fold_boundaries,
    region_artifact_paths,
)

TRAVEL = "data/preprocessed/travel_times.parquet"
DWELL = "data/preprocessed/dwell_times.parquet"
TOLERANCE = 1e-6  # GPU mixed precision may change the last digits


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def newest_attempt(run: Path):
    """The newest attempt with a metrics.json, and the artifacts folder its command used."""
    attempts = sorted((p.parent for p in run.glob("attempt_*/metrics.json")),
                      key=lambda p: int(p.name.split("_")[1]))
    if not attempts:
        sys.exit(f"{run}: no attempt with metrics.json")
    attempt = attempts[-1]
    for line in (run / "summary.md").read_text().splitlines():
        out = re.search(r"--output-dir (\S+?)`?$", line)
        # Match on the attempt name only: L0/run1_seed42 was renamed after it ran
        if line.startswith("- Command:") and out and Path(out.group(1)).name == attempt.name:
            return attempt, re.search(r"--artifacts-dir (\S+)", line).group(1)
    sys.exit(f"{run}: no command for {attempt.name} in summary.md")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="run folder, e.g. data/runs/L0/run1_seed42")
    a = p.parse_args()
    attempt, artifacts = newest_attempt(a.run)
    stored = json.loads((attempt / "metrics.json").read_text())["folds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    say(f"{a.run}: {attempt.name}, artifacts {artifacts}, {len(stored)} folds, {device}")

    # Settings, regions and feature statistics as saved with the models
    first = torch.load(attempt / "fold_0.pt", map_location="cpu", weights_only=False)
    hyper = first["hyperparams"]
    # Models saved before a feature switch existed lack it; it was off for them (the default)
    missing = {k: v for k, v in FEATURE_CONFIG.items() if k not in hyper}
    assert not any(missing.values()), missing
    hyper = {**hyper, **missing}
    ratio_target = uses_ratio_target(hyper)
    regions = {g: first[f"{g}_regions"] for g in ("train", "val", "test")}
    found = assign_regions_to_splits(discover_regions(artifacts, observation_municipalities(TRAVEL, DWELL)))
    assert tuple(regions.values()) == found, "regions differ from the saved model"

    graphs, link_nodes, stop_nodes = {}, {}, {}
    for g in ("val", "test"):
        graphs[g], link_nodes[g], stop_nodes[g], _ = build_hetero_graph(
            *region_artifact_paths(regions[g], artifacts), feature_config=hyper,
            feature_norm_stats=first["feature_norm_stats"])
    freeflow = {g: dict(zip(n["id"].astype(str), n["osm_travel_time"].astype(float)))
                for g, n in link_nodes.items()}

    # Fold boundaries from the first and last observation of all groups, as in training
    extremes = []
    for path in (TRAVEL, DWELL):
        for g in ("train", "val", "test"):
            times = _read_municipalities(path, ["from_time"], regions[g])["from_time"]
            extremes += [times.min(), times.max()]
    time_range = pd.DataFrame({"from_time": pd.Series(extremes, dtype=times.dtype)})
    boundaries = fold_boundaries(time_range, time_range)
    assert len(boundaries) == len(stored)
    travel, dwell = load_observation_groups(TRAVEL, DWELL, {"val": regions["val"], "test": regions["test"]})
    say("data loaded")

    rows = []
    for fold, (train_start, train_end, val_end, test_end) in enumerate(boundaries):
        model, ck = load_checkpoint(attempt / f"fold_{fold}.pt", device)
        assert ck["fold"] == fold
        scalers = types.SimpleNamespace(link_scaler=ck["link_scaler"], stop_scaler=ck["stop_scaler"])
        data = FoldData(travel, dwell, link_nodes, stop_nodes, freeflow,
                        periods={"train": (train_start, train_end), "val": (train_end, val_end),
                                 "test": (val_end, test_end)},
                        ratio_target=ratio_target, log_transform=hyper["log-transform"])
        for subset in ("val_spatial", "test_spatial"):
            dataset = data.dataset(subset, fitted=scalers)
            result = evaluate(model, graphs[SUBSETS[subset][0]], dataset, device, hyper,
                              ck["link_scaler"], ck["stop_scaler"], ratio_target)
            df_eval, df_history = data.observations(subset)["link"]
            cities = dataset.link_municipalities
            naive = seasonal_naive_errors_by_region(df_eval, df_history, "link", np.unique(cities))
            scores = {}
            for city in np.unique(cities):
                mask = cities == city
                err = result.link_preds[mask] - result.link_targets[mask]
                s = _scale_by_naive(float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2))), naive[city])
                scores[city] = s
                rows.append({"subset": subset, "fold": fold + 1, "city": city, "observations": int(mask.sum()),
                             "mase": s.get("mase", ""), "rmsse": s.get("rmsse", "")})
            for metric in ("mase", "rmsse"):
                values = [s[metric] for s in scores.values() if metric in s]
                mine = float(np.exp(np.mean(np.log(values))))
                theirs = stored[fold][f"{subset}/link/{metric}"]
                if abs(mine - theirs) > TOLERANCE:
                    sys.exit(f"CHECK FAILED fold {fold + 1} {subset} {metric}: {mine:.6f} here, {theirs:.6f} in metrics.json")
            say(f"fold {fold + 1} {subset}: {len(scores)} cities, link RMSSE {mine:.4f} = metrics.json {theirs:.4f}")
            del dataset, result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = a.run / "city_scores.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    say(f"CHECK OK: every fold matches metrics.json; written {out}")


if __name__ == "__main__":
    main()
