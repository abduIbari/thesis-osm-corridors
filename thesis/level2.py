"""Spec step 2, level 2: OSM tag counts in each link's corridor, shrunk with PCA.

Every object with tags counts once for each corridor it lies in or touches:
a point when it lies in the corridor, a line or area when any part of it
touches the corridor (decided 2026-11-15; level 1 POIs keep Horn's centre
rule). Relations that are not areas (bus routes, turn restrictions, stop
areas) have no single place and are skipped. So are municipality, town and
district borders (boundary=administrative or boundary=place): they touch nearly
every corridor of their city, so their tags (population=..., admin_level=...)
only say which city a link is in. Tags that describe the map entry instead of
the place are dropped first (`map_entry_key`).

Run from ~/thesis after corridors.py cut and build:

    PY=bus-embeddings/thesis/.venv/bin/python
    T=bus-embeddings-l2/thesis
    $PY -I $T/level2.py check GM0503     # reader against level 1's POI counts
    $PY -I $T/level2.py count GM0503     # tag counts of one city
    $PY -I $T/level2.py tags             # pool the 8 training cities
    $PY -I $T/level2.py context          # what kind of object carries each candidate
    $PY -I $T/level2.py build --top 100 --name jev  # the level 2 columns (also --name all, --top 500)

`count` writes data/corridor/features/<GM>/level2_counts.csv (id, tag, count;
non-zero rows only) and level2_dropped.csv (key, objects). `tags` writes
data/corridor/level2/1_candidates/training_tags.csv and dropped_keys.csv. `context` writes
1_candidates/tag_context.csv, the input for the language model filter. `build` takes the
`--top` kept tags found on the most training links, turns their counts into
per-km values, and writes data/corridor/features/<GM>/level2_top<N>_<name>.csv (64
columns) for every counted city, plus what it learned on the training cities
in data/corridor/level2/4_builds/top<N>_<name>/. The kept tags come from
data/corridor/level2/3_kept/, written after the OpenJev filter.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import osmium
import shapely
from shapely import STRtree
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))  # -I leaves the script folder out
import level1  # noqa: E402  (corridors, metres, length floor, Horn's POI rule)

DATA = level1.DATA
L2 = DATA / "corridor/level2"  # 1_candidates, 2_openjev, 3_kept, 4_builds: see its README.md
TRAINING_CITIES = ["GM0281", "GM0503", "GM0546", "GM0590", "GM0599", "GM0629", "GM1681", "GM1950"]
CANDIDATE_SHARE = 0.01  # a candidate is on at least 1% of training links
CANDIDATE_CITIES = 2  # ... and in at least 2 of the 8 training cities
BORDERS = {"administrative", "place"}  # boundary=* values of skipped objects
BATCH = 100_000  # objects turned into shapes at a time, to keep memory low
TOP_TAGS = 100  # default number of kept tags that become level 2 inputs
PCA_SIZE = 64  # numbers per link after PCA, the same width as level 3


def map_entry_key(key):
    """True for keys that describe the map entry instead of the place (spec step 2, level 2.1).

    The spec's rule, read as: names (name, *_name, name:*, *_name:*), addresses
    (addr:*), sources and notes (source*, note*, fixme, description and its
    language variants), references and ids (ref*, *:ref, wikidata, wikipedia and
    their *:wikidata, *:wikipedia variants), contact and web (website, phone,
    email, contact:*), opening hours (opening_hours*), editing data
    (check_date*, survey*), and dates (start_date, *:date; added 2026-11-15:
    every building year was its own tag, 171 of them among the candidates).
    Widened 2026-11-18 for keys with an extra part that slipped through (44
    candidates, mostly brand:website=...): *:name, *:website, *:phone, *:email,
    phone*, email*, website*, image* and created_by (the editing program).
    """
    k = key.lower()
    return (
        k == "name" or k.startswith("name:") or k.endswith("_name") or "_name:" in k
        or k.endswith((":name", ":website", ":phone", ":email")) or ":name:" in k
        or k.startswith(("phone", "email", "website", "image")) or k == "created_by"
        or k.startswith(("addr:", "source", "note", "ref", "contact:", "opening_hours", "check_date", "survey"))
        or k in ("fixme", "description", "website", "phone", "email", "wikidata", "wikipedia")
        or k.startswith("description:")
        or k.endswith((":ref", ":wikidata", ":wikipedia", ":date"))
        or k == "start_date"
    )


def read_objects(city, stats):
    """Every object of a city piece that has tags, in batches of (osm type, id, tags, WKB in lon/lat).

    Shapes follow level1.read_osm (osmnx's rules): a node is a point, a closed
    way is an area unless tagged area=no, an open way is a line, a multipolygon
    or boundary relation is an area. Other relations are skipped. Batches keep
    memory low; only closed ways wait until the end, in case osmium can't make
    an area of them.
    """
    wkb = osmium.geom.WKBFactory()
    reader = (
        osmium.FileProcessor(str(DATA / f"osm/cities/{city}.osm.pbf"))
        .with_locations()
        .with_areas()
        .with_filter(osmium.filter.EmptyTagFilter())
    )
    batch, closed_ways, ways_with_area = [], {}, set()
    for obj in reader:
        if obj.tags.get("boundary") in BORDERS:
            if obj.is_node() or obj.is_area() or (obj.is_way() and not obj.is_closed()):
                stats["skipped: border (boundary=administrative/place)"] += 1
            continue
        try:
            if obj.is_node():
                batch.append(("node", obj.id, {t.k: t.v for t in obj.tags}, wkb.create_point(obj)))
            elif obj.is_way():
                tags = {t.k: t.v for t in obj.tags}
                if obj.is_closed() and tags.get("area") != "no":
                    closed_ways[obj.id] = (tags, wkb.create_linestring(obj))
                else:
                    batch.append(("way", obj.id, tags, wkb.create_linestring(obj)))
            elif obj.is_area():
                if obj.from_way():
                    ways_with_area.add(obj.orig_id())
                    closed_ways.pop(obj.orig_id(), None)
                kind = "way" if obj.from_way() else "relation"
                batch.append((kind, obj.orig_id(), {t.k: t.v for t in obj.tags}, wkb.create_multipolygon(obj)))
            elif obj.is_relation() and obj.tags.get("type") not in ("multipolygon", "boundary"):
                stats["skipped: relation that is not an area"] += 1
        except RuntimeError:
            stats["skipped: no shape (missing node positions)"] += 1
        if len(batch) >= BATCH:
            stats["tagged objects"] += len(batch)
            yield batch
            batch = []
    # Closed ways that osmium could not turn into an area still get a polygon.
    for way_id, (tags, line_wkb) in closed_ways.items():
        if way_id in ways_with_area:
            continue
        line = shapely.from_wkb(line_wkb)
        if len(line.coords) >= 4:
            batch.append(("way", way_id, tags, shapely.make_valid(shapely.Polygon(line.coords)).wkb))
            stats["repaired closed ways (kept)"] += 1
        else:
            stats["skipped: closed way with under 4 points"] += 1
    stats["tagged objects"] += len(batch)
    yield batch


def report(city, stats):
    print(f"{city}: " + ", ".join(f"{k} {v}" for k, v in stats.items()))


def shapes_in_metres(objects):
    """Shapes of a slice of read_objects' output, in metres (EPSG:28992)."""
    return level1.to_metres(np.array([shapely.from_wkb(o[3]) for o in objects], dtype=object))


def count(city):
    ids, corridors, lengths = level1.load_corridors(city)
    tree = STRtree(corridors)
    stats = Counter()

    counts = defaultdict(Counter)  # link index -> tag -> objects
    dropped = Counter()  # key -> objects in the piece that carry it
    counted_objects = 0
    for part in read_objects(city, stats):
        obj_idx, link_idx = tree.query(shapes_in_metres(part), predicate="intersects")
        kept = []
        for kind, osm_id, tags, _ in part:
            for key in tags:
                if map_entry_key(key):
                    dropped[key] += 1
            kept.append([f"{k}={v}" for k, v in tags.items() if not map_entry_key(k)])
        counted_objects += len(set(obj_idx.tolist()))
        for o, link in zip(obj_idx, link_idx):
            counts[link].update(kept[o])

    out = DATA / f"corridor/features/{city}"
    out.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(out / "level2_counts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "tag", "count"])
        for link in range(len(ids)):
            for tag, n in sorted(counts[link].items()):
                writer.writerow([ids[link], tag, n])
                rows += 1
    with open(out / "level2_dropped.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "objects"])
        writer.writerows(dropped.most_common())

    tags_per_link = np.array([len(counts[i]) for i in range(len(ids))])
    distinct = len({t for c in counts.values() for t in c})
    print(f"wrote {out}/level2_counts.csv: {len(ids)} links, {rows} rows, {distinct} distinct tags")
    report(city, stats)
    print(f"  objects touching a corridor: {counted_objects} of {stats['tagged objects']}")
    print(f"  distinct tags per link: median {np.median(tags_per_link):.0f}, min {tags_per_link.min()}, max {tags_per_link.max()}")
    print(f"  links with no tag at all: {(tags_per_link == 0).sum()}")


def check(city):
    """Level 1's POI rule on this reader's objects must give level1.csv's counts exactly.

    Same piece, same shapes, Horn's rule: any object with shop or amenity is a
    POI, counted at its centre point. A difference means this reader sees other
    objects or shapes than level1.py.
    """
    ids, corridors, lengths = level1.load_corridors(city)
    stats = Counter()
    objects = [o for part in read_objects(city, stats) for o in part if "shop" in o[2] or "amenity" in o[2]]
    report(city, stats)
    centres = shapely.centroid(shapes_in_metres(objects))
    pois = [(kind, osm_id, level1.poi_category(tags), c) for (kind, osm_id, tags, _), c in zip(objects, centres)]
    ours = level1.poi_counts(corridors, pois)

    km = np.maximum(lengths, level1.MIN_LENGTH_M) / 1000
    with open(DATA / f"corridor/features/{city}/level1.csv", newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    worst = 0.0
    for name, col in ours.items():
        theirs = np.array([float(rows[i][f"corridor_poi_{name}_per_km"]) for i in ids]) * km
        worst = max(worst, float(np.abs(col - theirs).max()))
    total = sum(col.sum() for col in ours.values())
    print(f"{city}: {len(objects)} shop/amenity objects, {total:.0f} counted in corridors (centre rule)")
    print(f"  largest per-link difference to level1.csv over 12 categories: {worst:.6f} POIs")
    print("  OK, same as level 1" if worst < 1e-6 else "  DIFFERENT from level 1")


def tags():
    """Pool the training cities: on how many training links does each tag appear?"""
    links_with = Counter()
    total = Counter()
    cities_with = defaultdict(set)
    dropped = Counter()
    n_links = 0
    for city in TRAINING_CITIES:
        folder = DATA / f"corridor/features/{city}"
        with open(DATA / f"corridor/geometry/{city}/links.csv", newline="") as f:
            n_links += sum(1 for _ in csv.DictReader(f))
        with open(folder / "level2_counts.csv", newline="") as f:
            for r in csv.DictReader(f):
                links_with[r["tag"]] += 1
                total[r["tag"]] += int(r["count"])
                cities_with[r["tag"]].add(city)
        with open(folder / "level2_dropped.csv", newline="") as f:
            for r in csv.DictReader(f):
                dropped[r["key"]] += int(r["objects"])

    out = L2 / "1_candidates"
    out.mkdir(parents=True, exist_ok=True)
    candidates = 0
    with open(out / "training_tags.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "links", "share_of_links", "total_count", "cities", "candidate"])
        for tag, links in sorted(links_with.items(), key=lambda kv: (-kv[1], kv[0])):
            is_candidate = links / n_links >= CANDIDATE_SHARE and len(cities_with[tag]) >= CANDIDATE_CITIES
            candidates += is_candidate
            writer.writerow([tag, links, f"{links / n_links:.4f}", total[tag], len(cities_with[tag]), int(is_candidate)])
    with open(out / "dropped_keys.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "objects"])
        writer.writerows(dropped.most_common())

    need = int(np.ceil(CANDIDATE_SHARE * n_links))
    print(f"{n_links} training links, {len(links_with)} distinct tags")
    print(f"candidates (on at least {need} links, 1%, and in at least {CANDIDATE_CITIES} cities): {candidates}")
    print(f"wrote {out}/training_tags.csv and dropped_keys.csv ({len(dropped)} dropped keys)")


# OSM's main feature keys (wiki "Map features"): they say what an object is. Used by `context`.
MAIN_KEYS = ["highway", "railway", "public_transport", "amenity", "shop", "building", "landuse", "leisure",
             "natural", "barrier", "man_made", "waterway", "water", "power", "tourism", "office", "craft",
             "emergency", "healthcare", "historic", "place", "aeroway", "aerialway", "military", "telecom",
             "sport", "route", "boundary"]


def main_tag(tags, skip):
    """What the object is: its first main-key tag other than `skip`, or None."""
    for key in MAIN_KEYS:
        if key in tags and f"{key}={tags[key]}" != skip:
            return f"{key}={tags[key]}"
    return None


def context():
    """For each candidate: which kind of object carries it, among the training objects that touch a corridor.

    A tag like bench=yes says little alone; "mostly on highway=bus_stop" says what it means. The language model
    filter gets this as context.
    """
    with open(L2 / "1_candidates/training_tags.csv", newline="") as f:
        candidates = {r["tag"] for r in csv.DictReader(f) if r["candidate"] == "1"}
    carriers = defaultdict(Counter)  # tag -> what the objects carrying it are
    for city in TRAINING_CITIES:
        ids, corridors, _ = level1.load_corridors(city)
        tree = STRtree(corridors)
        stats = Counter()
        for part in read_objects(city, stats):
            touching = np.unique(tree.query(shapes_in_metres(part), predicate="intersects")[0])
            for o in touching:
                tags = part[o][2]
                for k, v in tags.items():
                    tag = f"{k}={v}"
                    if tag in candidates:
                        carriers[tag][main_tag(tags, tag) or "(nothing else)"] += 1
        print(f"{city}: done")

    out = L2 / "1_candidates/tag_context.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "objects", "carrier_1", "share_1", "carrier_2", "share_2", "carrier_3", "share_3"])
        for tag in sorted(candidates):
            n = sum(carriers[tag].values())
            top = [(c, k / n) for c, k in carriers[tag].most_common(3)] if n else []
            top += [("", 0.0)] * (3 - len(top))
            writer.writerow([tag, n, *(x for c, s in top for x in (c, f"{s:.2f}"))])
    print(f"wrote {out}: {len(candidates)} candidates")


def per_km_table(city, chosen):
    """Link ids of a city and its (links x tags) table of counts per km, zeros where a tag is absent."""
    ids, _, lengths = level1.load_corridors(city)
    row = {link_id: i for i, link_id in enumerate(ids)}
    col = {tag: j for j, tag in enumerate(chosen)}
    table = np.zeros((len(ids), len(chosen)))
    with open(DATA / f"corridor/features/{city}/level2_counts.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["tag"] in col:
                table[row[r["id"]], col[r["tag"]]] = int(r["count"])
    km = np.maximum(lengths, level1.MIN_LENGTH_M) / 1000
    return ids, table / km[:, None]


def build(top, name, kept_file, out):
    """Level 2 columns: top kept tags, per km, log1p, z-score, PCA. All learned on the training cities.

    `name` is the kept list (3_kept/<name>.csv: jev or all), so the version is called top<N>_<name>. What the
    build prints also goes to 4_builds/top<N>_<name>/build.log.
    """
    import datetime
    version = f"top{top}_{name}"
    kept_file = kept_file or L2 / f"3_kept/{name}.csv"
    if not kept_file.exists():
        raise SystemExit(f"{kept_file} is missing: write it with jev_score.py kept")
    lines = [f"level2.py build --top {top} --name {name}, {datetime.datetime.now():%Y-%m-%d %H:%M}",
             f"kept list: {kept_file}"]

    def say(text):
        print(text)
        lines.append(text)

    with open(kept_file, newline="") as f:
        kept = {r["tag"] for r in csv.DictReader(f)}
    with open(L2 / "1_candidates/training_tags.csv", newline="") as f:  # sorted by links, most first
        ranked = [r for r in csv.DictReader(f) if r["candidate"] == "1" and r["tag"] in kept]
    chosen = [r["tag"] for r in ranked[:top]]
    if len(chosen) < top:
        raise SystemExit(f"only {len(chosen)} kept candidate tags, fewer than --top {top}")

    cities = sorted(p.parent.name for p in (DATA / "corridor/features").glob("*/level2_counts.csv"))
    missing = [c for c in TRAINING_CITIES if c not in cities]
    if missing:
        raise SystemExit(f"run count first for the training cities {missing}")
    tables = {c: per_km_table(c, chosen) for c in cities}

    # Learn the scaling and the PCA on the training links only.
    train = np.log1p(np.vstack([tables[c][1] for c in TRAINING_CITIES]))
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std == 0] = 1.0
    size = min(PCA_SIZE, len(chosen))
    # scikit-learn's PCA (switched from plain numpy SVD on 2026-11-20, same result). svd_solver="full": the
    # default picks a randomized solver for wide tables like the top 500, which would make reruns differ.
    pca = PCA(n_components=size, svd_solver="full").fit((train - mean) / std)
    directions = pca.components_.copy()
    # A direction and its opposite are the same PCA result; fix the sign so reruns agree.
    directions *= np.sign(directions[np.arange(size), np.abs(directions).argmax(axis=1)])[:, None]
    explained = pca.explained_variance_ratio_

    fit = out / f"level2/4_builds/{version}"
    fit.mkdir(parents=True, exist_ok=True)
    with open(fit / "tags.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "training_links", "mean_log1p_per_km", "std_log1p_per_km"])
        for r, m, sd in zip(ranked, mean, std):
            writer.writerow([r["tag"], r["links"], repr(float(m)), repr(float(sd))])
    with open(fit / "pca_directions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", *chosen])
        for i, d in enumerate(directions, 1):
            writer.writerow([i, *(repr(float(v)) for v in d)])
    with open(fit / "explained_variance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "share", "running_total"])
        for i, (e, total) in enumerate(zip(explained[:size], np.cumsum(explained)[:size]), 1):
            writer.writerow([i, f"{e:.6f}", f"{total:.6f}"])

    names = [f"corridor_l2_pca_{i:02d}" for i in range(1, size + 1)]
    for city, (ids, table) in tables.items():
        values = ((np.log1p(table) - mean) / std) @ directions.T
        folder = out / f"features/{city}"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"level2_{version}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", *names])
            for link_id, row in zip(ids, values):
                writer.writerow([link_id, *(repr(float(v)) for v in row)])

    others = [c for c in cities if c not in TRAINING_CITIES]
    say(f"top {top} of {len(ranked)} kept candidate tags; the last one is on {ranked[top - 1]['links']} training links")
    say(f"learned on {len(train)} training links; {size} PCA numbers keep {explained[:size].sum():.1%} of the variation")
    say(f"  first 5 numbers: " + ", ".join(f"{e:.1%}" for e in explained[:5]))
    say(f"  number 64 alone: {explained[size - 1]:.2%}")
    say(f"wrote level2_{version}.csv for {len(cities)} cities ({len(TRAINING_CITIES)} training, {len(others)} others: {' '.join(others)})")
    say(f"wrote {fit}/tags.csv, pca_directions.csv, explained_variance.csv, build.log")
    (fit / "build.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["count", "check", "tags", "context", "build"])
    parser.add_argument("city", nargs="?")
    parser.add_argument("--top", type=int, default=TOP_TAGS, help="build: number of kept tags to use")
    parser.add_argument("--name", choices=["jev", "all"], help="build: which kept list, 3_kept/<name>.csv")
    parser.add_argument("--kept", type=Path, help="build: another csv with a `tag` column instead of 3_kept/<name>.csv")
    parser.add_argument("--out", type=Path, default=DATA / "corridor", help="build: where the output goes")
    args = parser.parse_args()
    if args.command == "tags":
        tags()
    elif args.command == "context":
        context()
    elif args.command == "build":
        if not args.name:
            parser.error("build needs --name jev or --name all")
        build(args.top, args.name, args.kept, args.out)
    else:
        if not args.city:
            parser.error("count and check need a city")
        {"count": count, "check": check}[args.command](args.city)
