#!/bin/sh
# Build the corridor features of the given cities, one city at a time:
# cut its OSM piece, draw the corridors, compute level 1, check against Horn.
# Run from ~/thesis. Each city's output and peak memory go to
# data/corridor/logs/<GM>.log.
#
#   sh bus-embeddings-corridor/thesis/build_corridor_features.sh GM0599 GM0518
set -e
PY=bus-embeddings-corridor/thesis/.venv/bin/python
T=bus-embeddings-corridor/thesis
mkdir -p data/corridor/logs
for city in "$@"; do
    log=data/corridor/logs/$city.log
    : > "$log"
    for step in "corridors.py cut" "corridors.py build" "level1.py build" "level1.py check"; do
        set -- $step
        echo "== $1 $2 $city" >> "$log"
        /usr/bin/time -f "peak %M KB, %e s" -a -o "$log" $PY -I "$T/$1" "$2" "$city" >> "$log" 2>&1
    done
    echo "$city done: $(grep -c '^peak' "$log") steps"
done
