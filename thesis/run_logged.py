"""Run one training run with logging, timing, a memory safety stop and retries.

Keeps everything: the full log, metrics.json and checkpoints of each attempt in
RUN_DIR/attempt_N/. Writes RUN_DIR/summary.md with the time per epoch, per fold
and in total, peak RAM, swap and GPU memory, and the reason of any failure.
Stops an attempt only if free RAM plus free swap drops below MIN_FREE_MB, and
retries a failed attempt up to MAX_ATTEMPTS times.

Usage: python run_logged.py RUN_DIR [train_embeddings arguments ...]
"""
import datetime as dt
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/abdulbari/thesis")
RUN_DIR = Path(sys.argv[1])
TRAIN_ARGS = sys.argv[2:]
MAX_ATTEMPTS = 3
MIN_FREE_MB = 1000
PYTHON = str(ROOT / "bus-embeddings/.venv/bin/python")

RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / "runner.pid").write_text(str(os.getpid()))
summary = open(RUN_DIR / "summary.md", "a")


def note(msg: str = "") -> None:
    summary.write(msg + "\n")
    summary.flush()


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def meminfo_mb(key: str) -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) // 1024
    return 0


def rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def shell(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=ROOT).stdout.strip()
    except Exception as e:
        return f"unavailable ({e})"


def gpu_mb() -> int:
    out = shell(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return int(out.splitlines()[0])
    except (ValueError, IndexError):
        return 0


def mins(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def run_attempt(n: int) -> bool:
    out_dir = RUN_DIR / f"attempt_{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, "-m", "bus_embeddings.cli.train_embeddings", *TRAIN_ARGS, "--output-dir", str(out_dir)]
    note(f"\n## Attempt {n}\n")
    note(f"- Started: {now()}")
    note(f"- Free at start: RAM {meminfo_mb('MemAvailable')} MB, swap {meminfo_mb('SwapFree')} MB, GPU used {gpu_mb()} MB")
    note(f"- Command: `{' '.join(cmd)}`")
    log = open(out_dir / "train.log", "w")
    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}, cwd=ROOT)
    state = {"peak_rss": 0, "peak_gpu": 0, "peak_swap": 0, "min_ram": 10**9, "stopped": "", "phase": "startup"}

    def sampler() -> None:
        tick = 0
        while proc.poll() is None:
            state["peak_rss"] = max(state["peak_rss"], rss_kb(proc.pid))
            ram, swap_free = meminfo_mb("MemAvailable"), meminfo_mb("SwapFree")
            state["min_ram"] = min(state["min_ram"], ram)
            state["peak_swap"] = max(state["peak_swap"], meminfo_mb("SwapTotal") - swap_free)
            if tick % 50 == 0:
                state["peak_gpu"] = max(state["peak_gpu"], gpu_mb())
            if ram + swap_free < MIN_FREE_MB:
                state["stopped"] = (f"safety stop during {state['phase']}: free RAM {ram} MB "
                                    f"+ free swap {swap_free} MB, below {MIN_FREE_MB} MB")
                proc.kill()
                return
            tick += 1
            time.sleep(0.2)

    threading.Thread(target=sampler, daemon=True).start()
    folds: list[dict] = []
    fold = None
    last_epoch = start
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        log.write(line)
        log.flush()
        t = time.time()
        tail = (tail + [line.rstrip()])[-40:]
        m = re.search(r"FOLD (\d+)/(\d+)", line)
        if m:
            fold = {"n": int(m.group(1)), "start": t, "epochs": [], "end": None, "early": None}
            folds.append(fold)
            last_epoch = t
            state["phase"] = f"fold {m.group(1)} setup"
            note(f"- {now()}: fold {m.group(1)}/{m.group(2)} started")
            continue
        m = re.search(r"Epoch (\d+) \|", line)
        if m and fold is not None:
            fold["epochs"].append(t - last_epoch)
            last_epoch = t
            state["phase"] = f"fold {fold['n']} epoch {int(m.group(1)) + 1}"
            continue
        m = re.search(r"Early stopping at epoch (\d+)", line)
        if m and fold is not None:
            fold["early"] = int(m.group(1))
            state["phase"] = f"fold {fold['n']} evaluation"
            continue
        if "Saved" in line and "fold_" in line and fold is not None:
            fold["end"] = t
            ep = fold["epochs"]
            note(f"- {now()}: fold {fold['n']} done in {mins(t - fold['start'])}, {len(ep)} epochs "
                 f"(mean {sum(ep) / max(len(ep), 1):.0f} s, first {ep[0] if ep else 0:.0f} s), "
                 f"{'early stop' if fold['early'] else 'all epochs used'}, "
                 f"evaluation {t - last_epoch:.0f} s")
            state["phase"] = "between folds"
    code = proc.wait()
    total = time.time() - start
    log.close()
    ok = code == 0 and (out_dir / "metrics.json").exists()
    note(f"- Finished: {now()}, exit code {code}, total {mins(total)}")
    note(f"- Peak RAM of training {state['peak_rss'] / 1048576:.2f} GB, lowest free RAM {state['min_ram']} MB, "
         f"swap used at peak {state['peak_swap']} MB, GPU memory at peak {state['peak_gpu']} MB")
    note("\n| Fold | Time | Epochs | Mean epoch | Early stop | Evaluation |")
    note("| --- | --- | --- | --- | --- | --- |")
    for f in folds:
        ep = f["epochs"]
        end = f["end"]
        evaluation = (end - f["start"] - sum(ep)) if end else None
        note(f"| {f['n']} | {mins(end - f['start']) if end else 'unfinished'} | {len(ep)} | "
             f"{sum(ep) / max(len(ep), 1):.0f} s | {f['early'] or '-'} | "
             f"{f'{evaluation:.0f} s' if evaluation is not None else '-'} |")
    if ok:
        note("\nResult: success. Scores are in metrics.json.")
        return True
    if state["stopped"]:
        reason = state["stopped"]
    elif code == -9 or code == 137:
        reason = "killed by the system (exit -9), most likely out of memory: check `dmesg` for oom-kill"
    elif any("Traceback" in l for l in tail):
        reason = "Python error, see the traceback below"
    else:
        reason = f"exit code {code}"
    note(f"\nResult: FAILED. Reason: {reason}")
    note("\nLast log lines:\n```")
    for l in tail[-25:]:
        note(l)
    note("```")
    return False


note(f"# Training run {RUN_DIR.name}\n")
note(f"- Runner started: {now()}")
note(f"- Code: {shell(['git', '-C', str(ROOT / 'bus-embeddings'), 'log', '--oneline', '-1'])} "
     f"(branch {shell(['git', '-C', str(ROOT / 'bus-embeddings'), 'branch', '--show-current'])})")
note(f"- Uncommitted code changes: {shell(['git', '-C', str(ROOT / 'bus-embeddings'), 'status', '--short', 'src']) or 'none'}")
note(f"- GPU: {shell(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'])}")
overall = time.time()
for attempt in range(1, MAX_ATTEMPTS + 1):
    if run_attempt(attempt):
        break
    if attempt < MAX_ATTEMPTS:
        note(f"\nRetrying in 60 s (attempt {attempt + 1} of {MAX_ATTEMPTS}).")
        time.sleep(60)
else:
    note(f"\nAll {MAX_ATTEMPTS} attempts failed.")
note(f"\n- Runner finished: {now()}, everything took {mins(time.time() - overall)}")
(RUN_DIR / "runner.pid").unlink(missing_ok=True)
