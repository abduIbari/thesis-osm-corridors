"""Experiment C: Horn's stop-circle features recounted with a smaller circle.

Horn's stop features (osm_poi_*_500m from graph/osm_features.py, osm_landuse_*
from graph/census_features.py) count POIs and land use in a 500 m circle around
each stop. Here they are recounted with Horn's own code, but from his cached
OSM downloads (data/cache/bus-embeddings/osmnx, fetched 8-19 May 2026) instead
of fresh ones, so only the radius changes and not the map:

  count   Fetch each city's 500 m area from the cache exactly as Horn's code does
          (the internet is blocked, so a cache miss stops the script), then run
          Horn's per-stop counting at 500 m and at the smaller radius. A smaller
          circle lies inside the 500 m area, so the cached data covers it.
  compare The 500 m recount must equal Horn's columns in data/artifacts.
  write   Copies of the three graph folders where only these 20 stop columns
          hold the smaller-radius values. Column names stay `..._500m` so
          Horn's code reads them unchanged.

Run from ~/thesis with the separate install that matches Horn's library
versions (requirements-circles.txt):

    PY=bus-embeddings/thesis/.venv-circles/bin/python
    $PY -I bus-embeddings/thesis/stop_circles.py count GM0503
    $PY -I bus-embeddings/thesis/stop_circles.py compare
    $PY -I bus-embeddings/thesis/stop_circles.py write
"""

import argparse
import csv
import math
import shutil
import socket
import sys
import time
import types
from pathlib import Path

DATA = Path("/home/abdulbari/thesis/data")
SRC = Path(__file__).parents[1] / "src"
CACHE = DATA / "cache/bus-embeddings/osmnx"
RADIUS = 200
OUT = DATA / f"circle{RADIUS}"
SOURCES = {"artifacts": DATA / "artifacts", "l1": DATA / "corridor/l1",
           "l2_top100_jev": DATA / "corridor/l2_top100_jev"}
CITIES = sorted(p.name for p in (DATA / "artifacts").iterdir() if p.name.startswith("GM"))

# Horn's modules, without the Valhalla client (not needed for stop features)
sys.path.insert(0, str(SRC))
sys.modules["bus_embeddings.graph.valhalla_router"] = types.SimpleNamespace(
    ValhallaRouter=None, valhalla_is_available=None)
from bus_embeddings.graph import census_features as horn_land  # noqa: E402
from bus_embeddings.graph import osm_features as horn_poi  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

COLUMNS = [f"osm_poi_{c}_500m" for c in horn_poi.POI_CATEGORY_NAMES] + \
          [f"osm_landuse_{c}" for c in horn_land.LANDUSE_CATEGORY_NAMES]


def offline(*args, **kwargs):
    raise RuntimeError("cache miss: osmnx tried to download; the recount must use Horn's cached data only")


def counts(city):
    """Horn's stop features of one city at 500 m and at RADIUS, from the cache only."""
    ox.settings.cache_folder = str(CACHE)
    ox.settings.use_cache = True
    requests.get = requests.post = requests.request = offline
    socket.create_connection = offline
    stops = pd.read_csv(DATA / "artifacts" / city / "nodes_stop.csv")
    real, fetched = ox.features_from_polygon, {}

    def recording(polygon, tags):  # Horn's 500 m query, answered by the cache
        key = tuple(sorted(tags))
        try:
            fetched[key] = real(polygon=polygon, tags=tags)
        except horn_poi.InsufficientResponseError as e:
            fetched[key] = e
        if isinstance(fetched[key], Exception):
            raise fetched[key]
        return fetched[key]

    def replay(polygon, tags):  # the same 500 m data for the smaller circle
        hit = fetched[tuple(sorted(tags))]
        if isinstance(hit, Exception):
            raise hit
        return hit

    out = {}
    for radius, fetch in ((500, recording), (RADIUS, replay)):
        ox.features_from_polygon = fetch
        poi = horn_poi.OSMFeatureExtractor.__new__(horn_poi.OSMFeatureExtractor)
        poi.poi_buffer_m, poi.projected_crs = float(radius), "EPSG:28992"
        land = horn_land.LanduseFeatureExtractor(buffer_m=float(radius), projected_crs="EPSG:28992")
        table = pd.concat([poi.extract_poi_features_for_stops(stops), land.extract_features(stops)], axis=1)
        table.insert(0, "id", stops["id"].values)
        out[radius] = table[["id"] + COLUMNS]
    ox.features_from_polygon = real
    return out


def count(cities):
    for city in cities:
        start = time.time()
        for radius, table in counts(city).items():
            path = OUT / f"counts/r{radius}/{city}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)
        print(f"{city}: {len(table)} stops, {time.time() - start:.0f} s", flush=True)


LANDUSE_TOLERANCE = 1e-9  # land-use shares (0 to 1) differ from Horn's in the 12th decimal: rounding in the area sums


def compare():
    """The 500 m recount against Horn's columns, every stop and column.

    POI counts must be exactly equal. Land-use shares may differ by at most
    LANDUSE_TOLERANCE (Delft: largest difference 2e-12)."""
    lines, ok = [], True
    poi = [c for c in COLUMNS if c.startswith("osm_poi_")]
    land = [c for c in COLUMNS if c.startswith("osm_landuse_")]
    for city in CITIES:
        horn = pd.read_csv(DATA / "artifacts" / city / "nodes_stop.csv")
        ours = pd.read_csv(OUT / f"counts/r500/{city}.csv")
        same_ids = list(horn["id"]) == list(ours["id"])
        poi_exact = bool((horn[poi].values == ours[poi].values).all())
        land_worst = float(abs(horn[land].values - ours[land].values).max())
        ok &= same_ids and poi_exact and land_worst <= LANDUSE_TOLERANCE
        lines.append(f"{city}: {len(horn)} stops, ids same {same_ids}, 12 POI columns exact {poi_exact}, "
                     f"land use largest difference {land_worst:.1e}")
    lines.append("ALL MATCH" if ok else "MISMATCH")
    (OUT / "counts/compare_r500.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return ok


def write():
    """Graph folder copies with the 20 stop columns at RADIUS; everything else copied unchanged."""
    if "ALL MATCH" not in (OUT / "counts/compare_r500.txt").read_text():
        sys.exit("compare has not passed; not writing")
    for city in CITIES:
        new = {r["id"]: r for r in csv.DictReader(open(OUT / f"counts/r{RADIUS}/{city}.csv", newline=""))}
        with open(DATA / "artifacts" / city / "nodes_stop.csv", newline="") as f:
            rows = list(csv.DictReader(f))
            header = list(rows[0])
        for row in rows:
            for c in COLUMNS:
                value = float(new[row["id"]][c])
                assert math.isfinite(value), (city, row["id"], c)
                row[c] = repr(value)
        for name, source in SOURCES.items():
            assert (source / city / "nodes_stop.csv").read_bytes() == (DATA / "artifacts" / city / "nodes_stop.csv").read_bytes()
            folder = OUT / name / city
            folder.mkdir(parents=True, exist_ok=True)
            for f in ("nodes_link.csv", "edges.csv"):
                shutil.copyfile(source / city / f, folder / f)
            with open(folder / "nodes_stop.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
        print(f"{city}: written to {', '.join(SOURCES)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("cities", nargs="*", default=CITIES)
    sub.add_parser("compare")
    sub.add_parser("write")
    a = p.parse_args()
    if a.cmd == "count":
        count(a.cities)
    elif a.cmd == "compare":
        sys.exit(0 if compare() else 1)
    else:
        write()
