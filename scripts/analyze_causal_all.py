#!/usr/bin/env python3
"""
Comprehensive DPS analysis for all three causal experiments:

  Exp 1: condition_on_previous_text toggle       (no_prev_text files)
  Exp 2: initial_prompt calibration              (prompt_sweep files)
  Exp 3: temperature sweep                       (temp_sweep files)

One script, prints everything you need to decide if the story holds.
"""
import pandas as pd, numpy as np, os
from rapidfuzz.distance import Levenshtein

BASE = "/Users/1uckyeom/Desktop/GermanDialect/results/swissdial"

# ── Shared DPS computation ──
def ncer(hyp, ref):
    if pd.isna(hyp) or pd.isna(ref): return 0.0
    hyp, ref = str(hyp).strip(), str(ref).strip()
    if not ref or not hyp: return 0.0
    d = Levenshtein.distance(hyp, ref)
    return max(0.0, 1.0 - d / len(ref))

def calc_dps(row, pred_col):
    ds = ncer(row[pred_col], str(row["ref_swiss_german"]))
    ns = ncer(row[pred_col], str(row["ref_high_german"]))
    if max(ds, ns) < 0.1: return None
    return ds - ns

SIZES_LABEL = {"tiny":"tiny","small":"small","medium":"medium","large_v3":"large-v3"}

# ═══════════════════════════════════════════════════════════════
# EXP 1: condition_on_previous_text
# ═══════════════════════════════════════════════════════════════
def exp1():
    base_cols = {"tiny":"whisper_tiny_text","small":"whisper_small_text",
                 "medium":"whisper_medium_text","large_v3":"whisper_large_v3_text"}
    intv_cols = {"tiny":"whisper_tiny_no_prev_text","small":"whisper_small_no_prev_text",
                 "medium":"whisper_medium_no_prev_text","large_v3":"whisper_large_v3_no_prev_text"}

    print("="*72)
    print("EXP 1: condition_on_previous_text  (True vs False)")
    print(f"{'Model':<16} {'Baseline':>10} {'NoPrevText':>12} {'Δ':>10}  Verdict")
    print("-"*72)
    for sk, label in [("tiny","tiny"),("small","small"),("medium","medium"),("large_v3","large-v3")]:
        bf = os.path.join(BASE, f"master_metadata_whisper_{sk}.csv")
        inf = os.path.join(BASE, f"master_metadata_whisper_{sk}_no_prev_text.csv")
        if not os.path.exists(inf): continue
        db = pd.read_csv(bf); di = pd.read_csv(inf)
        dps_b = [x for x in (calc_dps(r, base_cols[sk]) for _,r in db.iterrows()) if x is not None]
        dps_i = [x for x in (calc_dps(r, intv_cols[sk]) for _,r in di.iterrows()) if x is not None]
        mb, mi = np.mean(dps_b), np.mean(dps_i)
        d = mi - mb
        v = "no effect" if abs(d) < 0.002 else ("↑ dialect" if d > 0 else "↓ standard")
        print(f"Whisper {label:<10} {mb:>+10.4f} {mi:>+12.4f} {d:>+10.4f}  {v}")

# ═══════════════════════════════════════════════════════════════
# EXP 2: initial_prompt calibration
# ═══════════════════════════════════════════════════════════════
def exp2():
    print("\n" + "="*72)
    print("EXP 2: initial_prompt calibration")
    print("  empty = no text prior    dialect = Swiss German prompt")
    print("  standard = High German prompt")
    print(f"{'Model':<16} {'empty':>10} {'dialect':>10} {'standard':>10} {'Δ(dia-std)':>12}  Verdict")
    print("-"*72)
    for sk, label in [("tiny","tiny"),("small","small"),("medium","medium"),("large_v3","large-v3")]:
        pf = os.path.join(BASE, f"master_metadata_whisper_{sk}_prompt_sweep.csv")
        if not os.path.exists(pf): continue
        dp = pd.read_csv(pf)
        base_key = sk.replace("-","_")
        c_empty    = f"whisper_{base_key}_prompt_empty"
        c_dialect  = f"whisper_{base_key}_prompt_dialect"
        c_standard = f"whisper_{base_key}_prompt_standard"
        dps_emp = [x for x in (calc_dps(r, c_empty) for _,r in dp.iterrows()) if x is not None]
        dps_dia = [x for x in (calc_dps(r, c_dialect) for _,r in dp.iterrows()) if x is not None]
        dps_std = [x for x in (calc_dps(r, c_standard) for _,r in dp.iterrows()) if x is not None]
        me, md, ms = np.mean(dps_emp), np.mean(dps_dia), np.mean(dps_std)
        delta = md - ms
        n = len(dps_emp)
        v = "YES: prompt variety moves DPS" if abs(delta) > 0.003 else "NO: decoder insensitive to prompt variety"
        print(f"Whisper {label:<10} {me:>+10.4f} {md:>+10.4f} {ms:>+10.4f} {delta:>+12.4f}  {v} (n={n})")

# ═══════════════════════════════════════════════════════════════
# EXP 3: Temperature sweep
# ═══════════════════════════════════════════════════════════════
def exp3():
    TEMPS = [0.0, 0.2, 0.5, 0.8, 1.0]
    print("\n" + "="*72)
    print("EXP 3: Temperature sweep  (DPS vs T)")
    header = f"{'Model':<16}"
    for t in TEMPS: header += f"  T={t}"
    header += f"  {'Δ(T1.0−T0.0)':>14}"
    print(header)
    print("-"*72)
    for sk, label in [("tiny","tiny"),("small","small"),("medium","medium"),("large_v3","large-v3")]:
        tf = os.path.join(BASE, f"master_metadata_whisper_{sk}_temp_sweep.csv")
        if not os.path.exists(tf): continue
        dt = pd.read_csv(tf)
        base_key = sk.replace("-","_")
        means = []
        for T in TEMPS:
            col = f"whisper_{base_key}_T{str(T).replace('.','_')}"
            dps_vals = [x for x in (calc_dps(r, col) for _,r in dt.iterrows()) if x is not None]
            means.append(np.mean(dps_vals))
        line = f"Whisper {label:<10}"
        for m in means: line += f"  {m:+.4f}"
        slope = means[-1] - means[0]
        line += f"  {slope:>+14.4f}"
        v = "↑ DPS with T" if slope > 0.002 else ("↓ DPS with T" if slope < -0.002 else "flat")
        line += f"  {v}"
        print(line)

# ── Main ──
if __name__ == "__main__":
    exp1()
    exp2()
    exp3()
    print("\n" + "="*72)
    print("STORY CHECK:")
    print("  Exp 1: Δ ≈ 0 for all sizes → short-audio toggle has no effect (expected)")
    print("  Exp 2: Δ(dia−std) > 0? → decoder PRIOR responds to prompt variety?")
    print("  Exp 3: DPS ↑ with T? → LM prior strength modulates dialect preservation?")
    print("  Cross-scale: does effect size grow with model size?")
