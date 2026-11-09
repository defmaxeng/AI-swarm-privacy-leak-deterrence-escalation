"""Phase 2 supervisor (deviation D4): every --interval seconds, relaunch any shard whose process has exited while some
of its cells are still unfinished (crashed after rate-limit retries, or judged-pending). A relaunch resumes crashed
episodes from their checkpoints with the same seed. Never starts a second process for a shard that is running.
Logs only counts (no outcomes) to logs/phase2_supervisor.out. Exits when every shard's cells are done and judged.

  python scripts/supervise_phase2.py
API keys are read from .env (ADAPTION_API_KEY, ADAPTION_API_KEY_2) and passed to each shard process's environment.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import prereg_params as pp  # noqa: E402

# (model, part or None, out folder, key env name in .env, workers)
SHARDS = [
    ("glm-5.3-flash", "0/2", "runs/phase2/shard_glm-5.3-flash", "ADAPTION_API_KEY", 5),
    ("glm-5.3-flash", "1/2", "runs/phase2/shard_glm-5.3-flash_p1", "ADAPTION_API_KEY_2", 3),
    ("gemma-4-26b-a4b", None, "runs/phase2/shard_gemma-4-26b-a4b", "ADAPTION_API_KEY", 2),
    ("qwen3.8-27b", None, "runs/phase2/shard_qwen3.8-27b", "ADAPTION_API_KEY_2", 3),
    ("deepseek-v4-flash", None, "runs/phase2/shard_deepseek-v4-flash", "ADAPTION_API_KEY_2", 2),
]
LOG = ROOT / "logs/phase2_supervisor.out"


def dotenv():
    out = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def expected(P, sel, model, part):
    T = pp.phase2_trials(P, sel["control_env"], sel["disclosure_env"], [model])
    if part:
        k, n = map(int, part.split("/"))
        T = [t for i, t in enumerate(T) if i % n == k]
    return {pp.cell_key(t) for t in T}


def done_keys(out):
    keys = set()
    f = ROOT / out / "episodes.jsonl"
    if f.exists():
        for line in open(f, encoding="utf-8"):
            try:
                e = json.loads(line)
                if not e.get("judge_pending"):
                    keys.add(e["key"])
            except json.JSONDecodeError:
                pass
    return keys


def running(model, part, out):
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "run_prereg.py" in cmd and "--phase 2" in cmd and f"--models {model}" in cmd and out in cmd.replace("\\", "/"):
            if (part is None and "--part" not in cmd) or (part and f"--part {part}" in cmd):
                return True
    return False


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    P = pp.load(verbose=False)
    sel = json.loads((ROOT / "results/phase1_selection.json").read_text(encoding="utf-8"))
    env = dotenv()
    exp = {s: expected(P, sel, s[0], s[1]) for s in SHARDS}
    log(f"supervisor started; checking every {interval}s")
    while True:
        status, all_done = [], True
        for s in SHARDS:
            model, part, out, keyname, workers = s
            n_done = len(done_keys(out) & exp[s])
            left = len(exp[s]) - n_done
            alive = running(model, part, out)
            tag = f"{model}{'[' + part + ']' if part else ''}"
            status.append(f"{tag} {n_done}/{len(exp[s])}{' run' if alive else ''}")
            if left:
                all_done = False
                if not alive:
                    cmd = [sys.executable, "src/run_prereg.py", "--phase", "2", "--models", model, "--out", out,
                           "--workers", str(workers)] + (["--part", part] if part else [])
                    logf = open(ROOT / f"logs/phase2_{Path(out).name.replace('shard_', '')}.out", "a", encoding="utf-8")
                    subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                     env={**os.environ, "ADAPTION_API_KEY": env[keyname]})
                    log(f"RELAUNCH {tag} ({left} cells left, key {keyname}, {workers} workers)")
        log(" | ".join(status))
        if all_done:
            log("ALL PHASE 2 CELLS DONE AND JUDGED")
            return
        time.sleep(interval)


if __name__ == "__main__":
    main()
