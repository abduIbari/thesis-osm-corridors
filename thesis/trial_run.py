"""Short trial of the real training, keeping only RAM, GPU memory and timing.

Runs train_embeddings for 1 epoch per fold. Its log, metrics and checkpoints go
to a run folder that is deleted at the end, so no score is kept or shown. Stops
the run only if free RAM plus free swap drops below MIN_FREE_MB.

Usage: python trial_run.py SUMMARY_FILE
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/abdulbari/thesis")
RUN_DIR = ROOT / "data/memcheck/trial_run_tmp"
SUMMARY = Path(sys.argv[1])
MIN_FREE_MB = 1000
SCORE_WORDS = re.compile(r"rmsse|mase|mae|rmse|f1|loss|score|mad", re.IGNORECASE)

shutil.rmtree(RUN_DIR, ignore_errors=True)
RUN_DIR.mkdir(parents=True)
out = open(SUMMARY, "w")


def say(msg: str) -> None:
    out.write(msg + "\n")
    out.flush()


def available_mb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def swap_mb() -> tuple[int, int]:
    """(used, free) swap in MB."""
    total = free = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("SwapTotal:"):
                total = int(line.split()[1])
            elif line.startswith("SwapFree:"):
                free = int(line.split()[1])
    return (total - free) // 1024, free // 1024


def rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def gpu_mb() -> int:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 0


cmd = [
    str(ROOT / "bus-embeddings/.venv/bin/python"), "-m", "bus_embeddings.cli.train_embeddings",
    "--artifacts-dir", str(ROOT / "data/artifacts"),
    "--travel-times-parquet", str(ROOT / "data/preprocessed/travel_times.parquet"),
    "--dwell-times-parquet", str(ROOT / "data/preprocessed/dwell_times.parquet"),
    "--model", "gat", "--seed", "42", "--set", "epochs=1",
    "--output-dir", str(RUN_DIR / "out"),
]
say(f"Start: MemAvailable {available_mb()} MB, GPU used {gpu_mb()} MB")
start = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"}, cwd=ROOT)
peak_rss = peak_gpu = peak_swap = 0
min_avail = 10**9
swap_start = swap_mb()[0]
stopped = ""
current = "startup"
phase_peak: dict[str, int] = {}


def sampler() -> None:
    global peak_rss, peak_gpu, peak_swap, min_avail, stopped
    tick = 0
    while proc.poll() is None:
        rss = rss_kb(proc.pid)
        peak_rss = max(peak_rss, rss)
        phase_peak[current] = max(phase_peak.get(current, 0), rss)
        avail = available_mb()
        min_avail = min(min_avail, avail)
        used_swap, free_swap = swap_mb()
        peak_swap = max(peak_swap, used_swap)
        if tick % 25 == 0:
            peak_gpu = max(peak_gpu, gpu_mb())
        if avail + free_swap < MIN_FREE_MB:
            stopped = f"stopped by the watchdog during '{current}': RAM {avail} MB + swap {free_swap} MB free"
            proc.kill()
            return
        tick += 1
        time.sleep(0.2)


threading.Thread(target=sampler, daemon=True).start()
tail: list[str] = []
last = start
assert proc.stdout is not None
for line in proc.stdout:
    now = time.time()
    if "FOLD " in line and "/" in line:
        m = re.search(r"FOLD (\d+)/(\d+)", line)
        if m:
            say(f"{now - start:8.0f} s  fold {m.group(1)}/{m.group(2)} starts")
            current = f"fold {m.group(1)}"
    elif re.search(r"Epoch 0*1\b", line):
        say(f"{now - start:8.0f} s  epoch finished ({now - last:.0f} s since the last marker)")
        current = current + " evaluation"
    elif "Saved" in line and "fold_" in line:
        say(f"{now - start:8.0f} s  fold saved")
    last = now if ("FOLD " in line or "Epoch" in line) else last
    # Keep error text only, never lines that could hold a score
    if not SCORE_WORDS.search(line):
        tail = (tail + [line.rstrip()])[-25:]
code = proc.wait()
elapsed = time.time() - start
say(f"Exit code: {code}. Total time: {elapsed / 60:.1f} min")
if stopped:
    say(stopped)
say(f"Peak RAM of the training process: {peak_rss / 1048576:.2f} GB")
say(f"Lowest free RAM during the run: {min_avail} MB")
say(f"Peak GPU memory (sampled): {peak_gpu} MB")
say(f"Swap used: {swap_start} MB at start, {peak_swap} MB at peak")
for name, kb in phase_peak.items():
    say(f"  peak in {name}: {kb / 1048576:.2f} GB")
if code != 0:
    say("Last log lines without score words:")
    for line in tail:
        say("  " + line)
shutil.rmtree(RUN_DIR, ignore_errors=True)
say("Run folder deleted")
