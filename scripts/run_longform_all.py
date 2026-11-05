#!/usr/bin/env python3
"""
Overnight driver: all dialects × all models × both conditions.

  python scripts/run_longform_all.py

Runs sequentially, skips completed (resume-safe), prints ETA.
"""
import subprocess, sys, time, os
from pathlib import Path

MODELS   = ["tiny", "small", "medium", "large-v3"]
DIALECTS = ["ag","be","bs","gr","lu","sg","vs","zh"]

LONGFORM_DIR = Path(__file__).resolve().parent.parent / "data1.1" / "longform"
SCRIPT = Path(__file__).resolve().parent / "run_longform_toggle.py"

# Build task list — only for dialects that have longform audio
tasks = []
for m in MODELS:
    for d in DIALECTS:
        if (LONGFORM_DIR / d / "audio.wav").exists():
            tasks.append((m, d))

print(f"Total tasks: {len(tasks)} (each runs True + False)")
print(f"Models: {MODELS}")
print(f"Dialects: {[d for d in DIALECTS if (LONGFORM_DIR / d / 'audio.wav').exists()]}")
print("=" * 60)

t_start = time.time()
ok = 0
fail = 0

for i, (m, d) in enumerate(tasks):
    elapsed = time.time() - t_start
    eta_total = (elapsed / max(i, 1)) * len(tasks) if i > 0 else 0
    eta_remain = max(0, eta_total - elapsed)

    print(f"\n[{i+1}/{len(tasks)}] model={m} dialect={d}  "
          f"(elapsed {elapsed/60:.0f}m, ETA {eta_remain/60:.0f}m)", flush=True)

    ret = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", m, "--dialect", d],
        capture_output=False,  # live output
    )
    if ret.returncode == 0:
        ok += 1
    else:
        fail += 1
        print(f"  [!] FAILED (exit code {ret.returncode})", flush=True)

print(f"\n{'='*60}")
print(f"DONE. {ok} ok, {fail} failed, {time.time()-t_start:.0f}s total")
