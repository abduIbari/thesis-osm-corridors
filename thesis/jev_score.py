"""Spec step 2, level 2, part 3: OpenJev scores each candidate tag as a junk filter.

Each candidate gets a short description (what the thesis counts, how common the tag is, what kind of object
carries it) and three yes/no statements that say "this tag could matter for bus travel time" in different
words. OpenJev returns p(yes) for each; the score is their average. Below 0.3 the tag is dropped, above 0.7
kept, and the middle goes to a hand check.

Runs with its own install (thesis/jev/.venv, OpenJev's package `wev-ai`) and only inside the fence of
thesis/jev/run_fenced.sh: no network, no access to ~/.ssh or ~/.config/gh. From ~/thesis:

    J=bus-embeddings-l2/thesis/jev
    O=data/corridor/level2/2_openjev
    sh $J/run_fenced.sh score --prompt 1 --only $O/hand_labels.csv --out $O/tests/v1.csv   # the 40-tag test
    sh $J/run_fenced.sh score --prompt 1 --out $O/scores/v1.csv                            # all candidates
    python3 -I bus-embeddings-l2/thesis/jev_score.py check $O/tests/v1.csv
    python3 -I bus-embeddings-l2/thesis/jev_score.py kept $O/scores/v1.csv      # 3_kept/jev.csv and all.csv

`check` and `kept` need no model: `check` compares a score file with the hand labels, `kept` writes the kept-tag lists.
"""

import argparse
import csv
import sys
from pathlib import Path

DATA = Path("/home/abdulbari/thesis/data")
JEV = DATA / "corridor/level2/2_openjev"  # hand_labels.csv, tests/ (40 tags), scores/ (all candidates)
CANDIDATES = DATA / "corridor/level2/1_candidates"
MODEL = DATA / "models/OpenJev-4B-36871324"  # alanhuangya/OpenJev-4B at commit 368713247ad2
TRAINING_LINKS = 2517
DROP_BELOW, KEEP_ABOVE = 0.3, 0.7

TASK = """We are choosing OpenStreetMap tags as inputs for a model that predicts how long a city bus takes between two consecutive stops in Dutch cities. For each stretch between two stops, we count the map objects with a given tag within 100 m of the bus route.
"""
TAG = """Tag: {tag}
Found near {links} of {total} bus stretches ({share:.0%}), in {cities} of 8 cities.
The objects with this tag are mostly: {carriers}."""
# Examples come only from the tune half of hand_labels.csv, so the check half stays a fair test.
EXAMPLES = """Tags that matter, for example: highway=traffic_signals (buses wait at the lights), maxspeed=30 (the speed limit), crossing=zebra (buses stop for people crossing), shop=supermarket and amenity=school (they bring people and traffic).
Tags that do not matter, for example: natural=tree and leaf_type=broadleaved (trees along the street), backrest=yes (a detail of benches), roof:shape=gabled (a detail of buildings), payment:debit_cards=yes (a detail of shops).
"""
FIRM = "Most tags clearly matter or clearly do not, so judge firmly.\n"

# 1: first draft, the best on the 2026-11-15 test. Version 2 (busy places named in "affects", "kind" asking
# about mapping detail) did worse: data/corridor/level2/2_openjev/tests/v2.*. 3 adds examples, 4 adds examples and FIRM.
PROMPTS = {1: TASK + TAG, 3: TASK + EXAMPLES + TAG, 4: TASK + EXAMPLES + FIRM + TAG}
PROMPT_VERSION = 1  # the default; pick another with --prompt
QUESTIONS = {
    "affects": {
        "type": "noul",
        "instructions": "How many objects with this tag are near a street says something about how fast a bus can drive there or how often it is held up.",
        "criteria": {
            "true": "Yes: it is about the road, traffic control, speed, or places and land use that bring people and traffic.",
            "false": "No: it has nothing to do with bus speed or delays."
        }
    },
    "kind": {
        "type": "noul",
        "instructions": "This tag describes a road, traffic, a building, land use or a place that attracts people, and not street furniture, a small detail of another object, or map bookkeeping.",
        "criteria": {
            "true": "It describes such a feature.",
            "false": "It is street furniture, a detail of another object, or bookkeeping."
        }
    },
    "planner": {
        "type": "noul",
        "instructions": "A transport planner estimating bus travel times on a street would want to know how many of these are along it.",
        "criteria": {
            "true": "Yes, a planner would want this count.",
            "false": "No, a planner would ignore it."
        }
    }
}


# Version 5 (2026-11-20): "kind" misfired (traffic_signals 0.10, tower:type 0.88), so it is replaced by the spec's
# original second question: does the count mostly show how carefully the area was mapped? A yes there means junk,
# so that answer enters the score as 1 - p(yes).
MAPPED = {
    "type": "noul",
    "instructions": "How many objects with this tag are on the map near a street mostly depends on how carefully volunteers mapped the area, not on what the street and its traffic are like.",
    "criteria": {
        "true": "Yes: the count mostly shows how carefully the area was mapped, like trees, benches or details of buildings.",
        "false": "No: the count shows something real about the street, its traffic or the places along it."
    }
}
QUESTION_SETS = {1: QUESTIONS, 3: QUESTIONS, 4: QUESTIONS,
                 5: {"affects": QUESTIONS["affects"], "mapped": MAPPED, "planner": QUESTIONS["planner"]}}
FLIPPED = {"mapped"}  # p(yes) of these means junk
PROMPTS[5] = PROMPTS[1]


def band(score):
    return "drop" if score < DROP_BELOW else "keep" if score > KEEP_ABOVE else "middle"


def describe_carriers(ctx):
    parts = []
    for k in (1, 2, 3):
        carrier, share = ctx[f"carrier_{k}"], float(ctx[f"share_{k}"])
        if not carrier or share < 0.05:
            continue
        name = "objects whose main tag is this one" if carrier == "(nothing else)" else carrier
        parts.append(f"{name} ({share:.0%})")
    return ", ".join(parts) or "unknown"


def candidates(only=None):
    with open(CANDIDATES / "training_tags.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["candidate"] == "1"]
    with open(CANDIDATES / "tag_context.csv", newline="") as f:
        ctx = {r["tag"]: r for r in csv.DictReader(f)}
    if only:
        with open(only, newline="") as f:
            wanted = {r["tag"] for r in csv.DictReader(f)}
        rows = [r for r in rows if r["tag"] in wanted]
    return [(r, ctx[r["tag"]]) for r in rows]


def state_text(row, ctx, version):
    return PROMPTS[version].format(tag=row["tag"], links=row["links"], total=TRAINING_LINKS, share=float(row["share_of_links"]),
                        cities=row["cities"], carriers=describe_carriers(ctx))


def score(only, out, version):
    import torch
    import wev

    m = wev.load(str(MODEL))
    print(f"loaded {MODEL.name} on {m.model.device}; GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB; prompt version {version}")
    todo = candidates(only)
    import json
    Path(out).with_suffix(".prompt.json").write_text(json.dumps(
        {"version": version, "model": str(MODEL), "state": PROMPTS[version], "questions": QUESTION_SETS[version],
         "flipped": sorted(FLIPPED & set(QUESTION_SETS[version]))}, indent=1))
    questions = QUESTION_SETS[version]
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "links", "cities", *questions, "score", "band", "state"])  # raw p(yes) per question
        for i, (row, ctx) in enumerate(todo, 1):
            state = state_text(row, ctx, version)
            answers = m.predict(state=state, questions=questions)["answers"]
            p = [answers[q]["noul"] for q in questions]
            s = sum(1 - a if q in FLIPPED else a for q, a in zip(questions, p)) / len(p)
            writer.writerow([row["tag"], row["links"], row["cities"], *p, f"{s:.4f}", band(s), state])
            if i % 200 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    print(f"wrote {out}; peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


def check(scores):
    with open(JEV / "hand_labels.csv", newline="") as f:
        labels = {r["tag"]: r for r in csv.DictReader(f)}
    with open(scores, newline="") as f:
        reader = csv.DictReader(f)
        rows = {r["tag"]: r for r in reader}
        qs = reader.fieldnames[3:reader.fieldnames.index("score")]
    for part in ("tune", "check"):
        print(f"\n{part} set")
        print(f"  {'tag':32s} {'label':5s} " + " ".join(f"{q:>7s}" for q in qs) + f" {'score':>6s}  result")
        wrong = middle = 0
        for tag, lab in labels.items():
            if lab["set"] != part:
                continue
            r = rows[tag]
            b = r["band"]
            if lab["label"] == "hard":
                result = b
            elif b == "middle":
                result, middle = "middle", middle + 1
            elif b == lab["label"]:
                result = "right"
            else:
                result, wrong = "WRONG", wrong + 1
            print(f"  {tag[:32]:32s} {lab['label']:5s} " + " ".join(f"{float(r[q]):7.2f}" for q in qs)
                  + f" {float(r['score']):6.2f}  {result}")
        print(f"  clear tags on the wrong side: {wrong} of 16; in the middle: {middle} of 16")


def kept(scores, cut):
    """The two kept-tag lists for level2.py build (decided 2026-11-20): `jev.csv`, the candidates scoring `cut` or
    more, and `all.csv`, every candidate (the no-filter version). Most common first, like training_tags.csv."""
    with open(CANDIDATES / "training_tags.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["candidate"] == "1"]  # sorted by links, most first
    with open(scores, newline="") as f:
        score = {r["tag"]: float(r["score"]) for r in csv.DictReader(f)}
    missing = [r["tag"] for r in rows if r["tag"] not in score]
    if missing:
        sys.exit(f"{len(missing)} candidates have no score in {scores}, e.g. {missing[:3]}")
    out = DATA / "corridor/level2/3_kept"
    out.mkdir(parents=True, exist_ok=True)
    for name, chosen in (("jev", [r for r in rows if score[r["tag"]] >= cut]), ("all", rows)):
        with open(out / f"{name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tag", "links", "cities", "score"])
            for r in chosen:
                writer.writerow([r["tag"], r["links"], r["cities"], f"{score[r['tag']]:.4f}"])
        print(f"wrote {out / name}.csv: {len(chosen)} of {len(rows)} candidates")
    tags = [r["tag"] for r in rows]
    jev = [t for t in tags if score[t] >= cut]
    for n in (100, 500):
        out_n = [t for t in tags[:n] if t not in jev[:n]]
        in_n = [t for t in jev[:n] if t not in tags[:n]]
        print(f"top {n}: {n - len(out_n)} tags the same in both lists; OpenJev swaps {len(out_n)} out and {len(in_n)} in")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["score", "check", "kept"])
    parser.add_argument("scores", nargs="?", help="check, kept: a score file written by score")
    parser.add_argument("--cut", type=float, default=DROP_BELOW, help="kept: lowest score that is kept")
    parser.add_argument("--only", help="score: csv with a `tag` column, score only these")
    parser.add_argument("--out", help="score: where to write the scores")
    parser.add_argument("--prompt", type=int, choices=sorted(PROMPTS), default=PROMPT_VERSION, help="score: prompt version")
    args = parser.parse_args()
    if args.command == "score":
        if not args.out:
            sys.exit("score needs --out")
        score(args.only, args.out, args.prompt)
    else:
        if not args.scores:
            sys.exit(f"{args.command} needs a score file")
        check(args.scores) if args.command == "check" else kept(args.scores, args.cut)
