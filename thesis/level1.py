"""Spec step 2, level 1: Horn's stop surroundings, measured in each link's corridor.

12 POI categories as counts per km of link and 8 land-use categories as shares
of the corridor area. The rules copy Horn's stop features (graph/osm_features.py
and graph/census_features.py); only the area changes from a 500 m circle to the
corridor.

Run from ~/thesis after corridors.py cut and build:

    PY=bus-embeddings-corridor/thesis/.venv/bin/python
    $PY -I bus-embeddings-corridor/thesis/level1.py check GM0503
    $PY -I bus-embeddings-corridor/thesis/level1.py build GM0503

`check` compares this reader with Horn's own stop columns and shows how much
overlapping land-use areas change the shares. `build` writes
data/corridor/features/<GM>/level1.csv and pois.csv.
"""

import argparse
import ast
import csv
import json
import math
from pathlib import Path

import numpy as np
import osmium
import shapely
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import shape

DATA = Path("/home/abdulbari/thesis/data")
HORN = Path(__file__).parents[1] / "src/bus_embeddings/graph"

MIN_LENGTH_M = 50  # shorter links count as 50 m for per-km values
STOP_RADIUS_M = 500  # Horn's circle around a stop, used by `check` only
OSM_MARGIN_M = 200  # as in corridors.py
# True counts each spot of ground once per category. False adds up all areas
# like Horn's stop features, so overlapping areas count twice.
MERGE_LANDUSE_OVERLAPS = False

TO_METRES = Transformer.from_crs(4326, 28992, always_xy=True).transform


def horn_constant(file_name, constant):
    """Read a category dict from Horn's source without importing his package."""
    tree = ast.parse((HORN / file_name).read_text())
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(getattr(t, "id", None) == constant for t in targets):
            return ast.literal_eval(node.value)
    raise KeyError(constant)


POI_CATEGORIES = horn_constant("osm_features.py", "POI_CATEGORIES")
LANDUSE_CATEGORIES = horn_constant("census_features.py", "LANDUSE_CATEGORIES")
POI_NAMES = list(POI_CATEGORIES)
LANDUSE_NAMES = list(LANDUSE_CATEGORIES) + ["other"]
AMENITY_TO_CATEGORY = {v: cat for cat, values in POI_CATEGORIES.items() for v in values}
LANDUSE_TO_CATEGORY = {v: cat for cat, values in LANDUSE_CATEGORIES.items() for v in values}


def poi_category(tags):
    """Horn's _classify_pois: any shop is shopping, else the amenity value decides."""
    if "shop" in tags:
        return "shopping"
    return AMENITY_TO_CATEGORY.get(tags["amenity"].lower(), "other_amenity")


def to_metres(geoms):
    return shapely.transform(geoms, lambda c: np.column_stack(TO_METRES(c[:, 0], c[:, 1])))


def read_osm(city):
    """Return POI centre points and land-use polygons of a city piece, in metres.

    Shapes follow osmnx, which Horn uses: a node is a point, a closed way is a
    polygon unless tagged area=no, an open way is a line, and a multipolygon or
    boundary relation is a polygon.
    """
    wkb = osmium.geom.WKBFactory()
    reader = (
        osmium.FileProcessor(str(DATA / f"osm/cities/{city}.osm.pbf"))
        .with_locations()
        .with_areas()
        .with_filter(osmium.filter.KeyFilter("amenity", "shop", "landuse"))
    )
    found = []  # (osm type, id, tags, geometry in lon/lat)
    closed_ways, ways_with_area = {}, set()
    for obj in reader:
        tags = {t.k: t.v for t in obj.tags}
        try:
            if obj.is_node():
                found.append(("node", obj.id, tags, shapely.from_wkb(wkb.create_point(obj))))
            elif obj.is_way():
                line = shapely.from_wkb(wkb.create_linestring(obj))
                if obj.is_closed() and tags.get("area") != "no":
                    closed_ways[obj.id] = (tags, line)
                else:
                    found.append(("way", obj.id, tags, line))
            elif obj.is_area():
                kind = "way" if obj.from_way() else "relation"
                if obj.from_way():
                    ways_with_area.add(obj.orig_id())
                found.append((kind, obj.orig_id(), tags, shapely.from_wkb(wkb.create_multipolygon(obj))))
        except RuntimeError:
            continue  # an object with missing node positions has no shape
    # Closed ways that osmium could not turn into an area still get a polygon.
    repaired = 0
    for way_id, (tags, line) in closed_ways.items():
        if way_id not in ways_with_area and len(line.coords) >= 4:
            found.append(("way", way_id, tags, shapely.make_valid(shapely.Polygon(line.coords))))
            repaired += 1

    geoms = to_metres(np.array([f[3] for f in found], dtype=object))
    pois, landuse = [], []
    for (kind, osm_id, tags, _), geom in zip(found, geoms):
        if "shop" in tags or "amenity" in tags:
            pois.append((kind, osm_id, poi_category(tags), geom.centroid))
        if "landuse" in tags and geom.area > 0:
            category = LANDUSE_TO_CATEGORY.get(tags["landuse"].lower(), "other")
            landuse.append((category, geom if geom.is_valid else shapely.make_valid(geom)))
    print(f"{city}: {len(pois)} POIs, {len(landuse)} land-use areas, {repaired} repaired closed ways")
    return pois, landuse


def poi_counts(areas, pois):
    """Number of POI centre points inside each area, per category."""
    counts = {name: np.zeros(len(areas)) for name in POI_NAMES}
    points = np.array([p[3] for p in pois], dtype=object)
    area_idx, point_idx = STRtree(points).query(areas, predicate="contains")
    for a, p in zip(area_idx, point_idx):
        counts[pois[p][2]][a] += 1
    return counts


def landuse_area(areas, landuse, merge_overlaps):
    """Square metres of each land-use category inside each area."""
    result = {}
    for name in LANDUSE_NAMES:
        polygons = np.array([g for cat, g in landuse if cat == name], dtype=object)
        total = np.zeros(len(areas))
        if len(polygons):
            area_idx, poly_idx = STRtree(polygons).query(areas, predicate="intersects")
            pieces = shapely.intersection(areas[area_idx], polygons[poly_idx])
            if merge_overlaps:
                for a in np.unique(area_idx):
                    total[a] = shapely.union_all(pieces[area_idx == a]).area
            else:
                np.add.at(total, area_idx, shapely.area(pieces))
        result[name] = total
    return result


def load_corridors(city):
    folder = DATA / f"corridor/geometry/{city}"
    with open(folder / "links.csv", newline="") as f:
        lengths = {r["id"]: float(r["path_length_m"]) for r in csv.DictReader(f)}
    features = json.loads((folder / "corridors.geojson").read_text())["features"]
    ids = [f["properties"]["id"] for f in features]
    corridors = np.array([shape(f["geometry"]) for f in features], dtype=object)
    return ids, corridors, np.array([lengths[i] for i in ids])


def build(city):
    ids, corridors, lengths = load_corridors(city)
    pois, landuse = read_osm(city)
    km = np.maximum(lengths, MIN_LENGTH_M) / 1000
    corridor_area = shapely.area(corridors)

    columns = {}
    for name, counts in poi_counts(corridors, pois).items():
        columns[f"corridor_poi_{name}_per_km"] = counts / km
    for name, area in landuse_area(corridors, landuse, MERGE_LANDUSE_OVERLAPS).items():
        columns[f"corridor_landuse_{name}"] = np.minimum(area / corridor_area, 1.0)

    out = DATA / f"corridor/features/{city}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "level1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", *columns])
        for i, link_id in enumerate(ids):
            writer.writerow([link_id, *(repr(float(col[i])) for col in columns.values())])
    with open(out / "pois.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["osm_type", "osm_id", "category", "x", "y"])
        for kind, osm_id, category, point in pois:
            writer.writerow([kind, osm_id, category, round(point.x, 1), round(point.y, 1)])

    print(f"wrote {out}/level1.csv: {len(ids)} links, {len(columns)} columns")
    for name, col in columns.items():
        print(f"  {name:42s} mean {col.mean():8.3f}  max {col.max():8.3f}  links above 0: {(col > 0).sum()}")


def check(city):
    """Compare with Horn's stop columns, and Horn's land-use sum with the merged way."""
    pois, landuse = read_osm(city)

    # Part 1: the same reader on 500 m circles around the stops.
    with open(DATA / f"artifacts/{city}/nodes_stop.csv", newline="") as f:
        stops = list(csv.DictReader(f))
    xy = [TO_METRES(float(s["lon"]), float(s["lat"])) for s in stops]
    ids, corridors, _ = load_corridors(city)
    paths = json.loads((DATA / f"corridor/geometry/{city}/paths.geojson").read_text())["features"]
    minx, miny, maxx, maxy = shapely.union_all([shape(f["geometry"]) for f in paths]).bounds
    reach = STOP_RADIUS_M - OSM_MARGIN_M  # a circle must stay inside the OSM piece
    inside = np.array([minx + reach <= x <= maxx - reach and miny + reach <= y <= maxy - reach for x, y in xy])
    circles = np.array([shapely.Point(p).buffer(STOP_RADIUS_M) for p in xy], dtype=object)
    print(f"\nstops: {len(stops)}, with their whole circle inside the OSM piece: {inside.sum()}")

    ours = {f"osm_poi_{n}_500m": c for n, c in poi_counts(circles, pois).items()}
    circle_area = math.pi * STOP_RADIUS_M**2
    for name, area in landuse_area(circles, landuse, False).items():
        ours[f"osm_landuse_{name}"] = np.minimum(area / circle_area, 1.0)
    print(f"{'column':34s} {'Horn mean':>10s} {'ours mean':>10s} {'corr':>6s} {'same':>9s}")
    for column, mine in ours.items():
        horn = np.array([float(s[column]) for s in stops])[inside]
        mine = mine[inside]
        corr = np.corrcoef(horn, mine)[0, 1] if horn.std() > 0 and mine.std() > 0 else float("nan")
        tolerance = 0.5 if "poi" in column else 0.01
        same = (np.abs(horn - mine) <= tolerance).sum()
        print(f"{column:34s} {horn.mean():10.3f} {mine.mean():10.3f} {corr:6.3f} {same:5d}/{len(horn)}")

    # Part 2: does counting overlapping land-use areas twice change the shares?
    corridor_area = shapely.area(corridors)
    added = landuse_area(corridors, landuse, False)
    merged = landuse_area(corridors, landuse, True)
    print(f"\nland-use share per link, Horn's sum against the merged way ({len(ids)} links)")
    print(f"{'category':12s} {'sum mean':>9s} {'merged':>9s} {'links differing':>16s} {'largest gap':>12s}")
    any_gap = np.zeros(len(ids), dtype=bool)
    for name in LANDUSE_NAMES:
        a = np.minimum(added[name] / corridor_area, 1.0)
        m = np.minimum(merged[name] / corridor_area, 1.0)
        gap = np.abs(a - m)
        any_gap |= gap > 0.01
        print(f"{name:12s} {a.mean():9.4f} {m.mean():9.4f} {(gap > 0.01).sum():16d} {gap.max():12.4f}")
    print(f"links where any category differs by more than 0.01 (one percentage point): {any_gap.sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "build"])
    parser.add_argument("city")
    args = parser.parse_args()
    {"check": check, "build": build}[args.command](args.city)
