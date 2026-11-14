"""Spec step 3: copy Horn's graph folder and add one level's corridor columns.

For every city, data/artifacts/<GM>/ is copied to data/corridor/<level>/<GM>/:
nodes_stop.csv and edges.csv unchanged, nodes_link.csv with the level's columns
appended at the end of every line, matched by link id. Horn's own text in each
line is kept byte for byte, so his columns can't change.

Run from ~/thesis after level1.py build has run for all 18 cities:

    bus-embeddings/thesis/.venv/bin/python -I bus-embeddings/thesis/add_columns.py l1

Checks, any failure stops the script before anything is written:
    - every city in REGION_SPLIT has an artifact folder and a feature file
    - link ids are unique, and the feature file holds exactly the graph's links
    - every new value is a finite number
    - the new file, read back, has Horn's rows in Horn's order with Horn's
      values, plus the new columns
"""

import argparse
import ast
import csv
import io
import math
import shutil
from pathlib import Path

DATA = Path("/home/abdulbari/thesis/data")
CONFIG = Path(__file__).parents[1] / "src/bus_embeddings/training/config.py"
FEATURE_FILE = {"l1": "level1.csv"}


def all_cities():
    """The 18 cities from Horn's REGION_SPLIT, read without importing his package."""
    for node in ast.parse(CONFIG.read_text()).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "REGION_SPLIT":
            split = ast.literal_eval(node.value)
            return [city for cities in split.values() for city in cities]
    raise KeyError("REGION_SPLIT")


def read_features(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = {row[0]: row[1:] for row in reader}
    assert header[0] == "id", path
    for link_id, values in rows.items():
        assert len(values) == len(header) - 1, (path, link_id)
        assert all(math.isfinite(float(v)) for v in values), (path, link_id)
    return header[1:], rows


def build_city(city, level):
    src = DATA / f"artifacts/{city}"
    columns, features = read_features(DATA / f"corridor/features/{city}/{FEATURE_FILE[level]}")

    lines = (src / "nodes_link.csv").read_text().splitlines(keepends=False)
    header, body = lines[0], lines[1:]
    horn_rows = list(csv.reader(io.StringIO("\n".join(lines))))
    horn_header, horn_body = horn_rows[0], horn_rows[1:]
    assert len(horn_body) == len(body), f"{city}: a quoted field spans lines"
    assert horn_header[0] == "id" and not set(columns) & set(horn_header), city
    ids = [row[0] for row in horn_body]
    assert len(set(ids)) == len(ids), f"{city}: duplicate link ids"
    assert set(ids) == set(features), f"{city}: links differ between graph and features"

    new_text = header + "," + ",".join(columns) + "\n"
    new_text += "".join(line + "," + ",".join(features[i]) + "\n" for line, i in zip(body, ids))

    # Read the new file back and compare with Horn's rows plus our values.
    back = list(csv.reader(io.StringIO(new_text)))
    assert back[0] == horn_header + columns, city
    for row, horn_row, link_id in zip(back[1:], horn_body, ids):
        assert row == horn_row + features[link_id], (city, link_id)
    return new_text, len(ids), len(columns)


def main(level):
    cities = all_cities()
    assert len(cities) == 18, cities
    built = {city: build_city(city, level) for city in cities}  # all checks first

    for city, (text, n_links, n_cols) in built.items():
        out = DATA / f"corridor/{level}/{city}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "nodes_link.csv").write_text(text)
        for name in ("nodes_stop.csv", "edges.csv"):
            shutil.copyfile(DATA / f"artifacts/{city}/{name}", out / name)
        print(f"{city}: {n_links} links, {n_cols} columns added")
    print(f"wrote data/corridor/{level}/ for {len(built)} cities, "
          f"{sum(b[1] for b in built.values())} links")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("level", choices=sorted(FEATURE_FILE))
    main(parser.parse_args().level)
