"""Spec step 2: cut one OSM piece per city and build each link's corridor.

Run from ~/thesis with -I, because the inputs are downloaded data:

    PY=bus-embeddings-corridor/thesis/.venv/bin/python
    $PY -I bus-embeddings-corridor/thesis/corridors.py cut GM0503
    $PY -I bus-embeddings-corridor/thesis/corridors.py build GM0503

Output of `build` in data/corridor/geometry/<GM>/, coordinates in EPSG:28992:
    links.csv          id, lengths, detour ratio, odd-route flag, corridor area
    paths.geojson      each link's Valhalla path
    corridors.geojson  each link's corridor
"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

from pyproj import Transformer
from shapely import LineString, Point, is_valid
from shapely.geometry import mapping

DATA = Path("/home/abdulbari/thesis/data")
VALHALLA = DATA / "cache/bus-embeddings/valhalla"
PBF = DATA / "osm/netherlands-260101.osm.pbf"
OSMIUM = Path(__file__).parent / ".venv/bin/osmium"

CORRIDOR_M = 100
OSM_MARGIN_M = 200
ODD_ROUTE_RATIO = 3

TO_METRES = Transformer.from_crs(4326, 28992, always_xy=True).transform
TO_LONLAT = Transformer.from_crs(28992, 4326, always_xy=True).transform


def cache_file(row):
    """File name as in Horn's ValhallaRouter._get_cache_key, with (lon, lat) coordinates."""
    key = "{:.6f},{:.6f}|{:.6f},{:.6f}|bus".format(
        float(row["from_lon"]), float(row["from_lat"]), float(row["to_lon"]), float(row["to_lat"])
    )
    return VALHALLA / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def load_links(city):
    """Return each link's graph row and its Valhalla path in metres."""
    with open(DATA / f"artifacts/{city}/nodes_link.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    links = []
    for row in rows:
        with open(cache_file(row)) as f:
            lonlat = json.load(f)["geometry"]
        xy = [TO_METRES(lon, lat) for lon, lat in lonlat]
        links.append((row, xy))
    return links


def cut(city):
    links = load_links(city)
    xs = [x for _, xy in links for x, _ in xy]
    ys = [y for _, xy in links for _, y in xy]
    # A straight metre-grid box maps to a slightly tilted lon/lat shape, so take
    # the outer lon/lat limits of its four corners.
    corners = [
        TO_LONLAT(x, y)
        for x in (min(xs) - OSM_MARGIN_M, max(xs) + OSM_MARGIN_M)
        for y in (min(ys) - OSM_MARGIN_M, max(ys) + OSM_MARGIN_M)
    ]
    lons, lats = zip(*corners)
    bbox = f"{min(lons):.6f},{min(lats):.6f},{max(lons):.6f},{max(lats):.6f}"
    out = DATA / f"osm/cities/{city}.osm.pbf"
    out.parent.mkdir(exist_ok=True)
    print(f"{city}: {len(links)} links, bbox {bbox}")
    subprocess.run(
        [str(OSMIUM), "extract", "-b", bbox, "--strategy", "complete_ways",
         "--overwrite", "-o", str(out), str(PBF)],
        check=True,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def feature(link_id, geom):
    return {"type": "Feature", "properties": {"id": link_id}, "geometry": mapping(geom)}


def write_geojson(path, features):
    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:28992"}},
        "features": features,
    }
    path.write_text(json.dumps(collection))


def build(city):
    links = load_links(city)
    out = DATA / f"corridor/geometry/{city}"
    out.mkdir(parents=True, exist_ok=True)
    table, paths, corridors = [], [], []
    for row, xy in links:
        line = LineString(xy) if len(set(xy)) >= 2 else None
        # A zero-length path has no line to follow, so its corridor is the
        # circle around its point (round ends, decided 2026-11-09).
        centre = line if line is not None else Point(xy[0])
        corridor = centre.buffer(CORRIDOR_M, cap_style="round", join_style="round")
        assert is_valid(corridor) and not corridor.is_empty, row["id"]

        length = line.length if line is not None else 0.0
        start = TO_METRES(float(row["from_lon"]), float(row["from_lat"]))
        end = TO_METRES(float(row["to_lon"]), float(row["to_lat"]))
        straight = math.dist(start, end)
        ratio = length / straight if straight > 0 else None
        # Largest possible area: a straight path plus a full circle.
        assert corridor.area <= 2 * CORRIDOR_M * length + math.pi * CORRIDOR_M**2 + 1, row["id"]

        table.append({
            "id": row["id"],
            "path_length_m": round(length, 1),
            "straight_m": round(straight, 1),
            "detour_ratio": round(ratio, 3) if ratio is not None else "",
            "odd_route": int(ratio is not None and ratio > ODD_ROUTE_RATIO),
            "corridor_area_m2": round(corridor.area),
            "osm_length": row["osm_length"],
        })
        paths.append(feature(row["id"], line if line is not None else Point(xy[0])))
        corridors.append(feature(row["id"], corridor))

    with open(out / "links.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    write_geojson(out / "paths.geojson", paths)
    write_geojson(out / "corridors.geojson", corridors)

    lengths = sorted(r["path_length_m"] for r in table)
    print(f"{city}: {len(table)} corridors written to {out}")
    print(f"odd-route links: {sum(r['odd_route'] for r in table)}")
    print(f"zero-length paths: {sum(l == 0 for l in lengths)}, under 50 m: {sum(l < 50 for l in lengths)}")
    print(f"path length m: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cut", "build"])
    parser.add_argument("city")
    args = parser.parse_args()
    {"cut": cut, "build": build}[args.command](args.city)
