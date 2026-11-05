#!/usr/bin/env python3
"""
Recompute ALL paper numbers with the `max(|y|,|r|)` denominator (paper Eq 1-2),
so the paper's formula and numbers finally agree. Also dedups whisper_small.

This is the "denominator unification" pass (choice A). It reproduces every table
and inline number in main.tex under the correct NED normalization.

Pure CPU, no inference.
"""
import os
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

BASE = "/Users/1uckyeom/Desktop/GermanDialect/results/swissdial"


def sim(hyp, ref):
    """1 - NED, with the paper's max(|y|,|r|) denominator."""
    if pd.isna(hyp) or pd.isna(ref):
        return 0.0
    hyp, ref = str(hyp).strip(), str(ref).strip()
    if not hyp or not ref:
        return 0.0
    d = Levenshtein.distance(hyp, ref)
    return max(0.0, 1.0 - d / max(len(hyp), len(ref)))


def dps(row, pred_col):
    ds = sim(row[pred_col], row["ref_swiss_german"])
    ns = sim(row[pred_col], row["ref_high_german"])
    if max(ds, ns) < 0.1:
        return None, ds, ns
    return ds - ns, ds, ns


def mean_dps(df, col):
    vals = [x[0] for x in (dps(r, col) for _, r in df.iterrows()) if x[0] is not None]
    return np.mean(vals)


def mean_components(df, col):
    ds = [sim(r[col], r["ref_swiss_german"]) for _, r in df.iterrows()]
    ns = [sim(r[col], r["ref_high_german"]) for _, r in df.iterrows()]
    return np.mean(ds), np.mean(ns)


def load(fname):
    df = pd.read_csv(os.path.join(BASE, fname), encoding="utf-8-sig")
    df = df.dropna(subset=["sentence_id"])
    df["sentence_id"] = df["sentence_id"].apply(lambda s: int(float(str(s).split("_")[0])))
    return df.drop_duplicates(subset=["sentence_id", "dialect"], keep="first")


MODELS = [
    ("wav2vec2 300M",   "master_metadata_wav2vec_300m.csv",    "pred_text_wav2vec_300m"),
    ("wav2vec2 1B",     "master_metadata_wav2vec_1b.csv",      "pred_text_wav2vec_1b"),
    ("Whisper tiny",     "master_metadata_whisper_tiny.csv",    "whisper_tiny_text"),
    ("Whisper small",    "master_metadata_whisper_small.csv",   "whisper_small_text"),
    ("Whisper medium",   "master_metadata_whisper_medium.csv",  "whisper_medium_text"),
    ("Whisper large-v3", "master_metadata_whisper_large_v3.csv","whisper_large_v3_text"),
    ("Qwen3-ASR 0.6B",   "master_metadata_qwen3_06b.csv",      "Qwen3-ASR-0.6B_text"),
    ("Qwen3-ASR 1.7B",   "master_metadata_qwen3_17b.csv",      "qwen3_17b_text"),
]

print("=" * 78)
print("TABLE 1 (arch) — max denominator, whisper_small deduped")
print("=" * 78)
print(f"{'Model':<18}{'DialectSim':>11}{'NormSim':>10}{'DPS':>9}")
for name, fname, col in MODELS:
    df = load(fname)
    mds, mns = mean_components(df, col)
    mdps = mean_dps(df, col)
    print(f"{name:<18}{mds:>+11.3f}{mns:>+10.3f}{mdps:>+9.3f}")

print("\n" + "=" * 78)
print("TABLE 2 (prompt calibration) — max denominator")
print("=" * 78)
print(f"{'Model':<14}{'Empty':>9}{'Dialect':>10}{'Standard':>11}{'Delta':>9}")
for sk, label in [("tiny", "tiny"), ("small", "small"), ("medium", "medium"), ("large_v3", "large-v3")]:
    pf = os.path.join(BASE, f"master_metadata_whisper_{sk}_prompt_sweep.csv")
    if not os.path.exists(pf):
        continue
    dp = load(pf)
    bk = sk.replace("-", "_")
    e = mean_dps(dp, f"whisper_{bk}_prompt_empty")
    d = mean_dps(dp, f"whisper_{bk}_prompt_dialect")
    s = mean_dps(dp, f"whisper_{bk}_prompt_standard")
    print(f"Whisper {label:<8}{e:>+9.3f}{d:>+10.3f}{s:>+11.3f}{d-s:>+9.3f}")

print("\n" + "=" * 78)
print("TEMPERATURE SWEEP (DPS at T=0.0 and T=1.0) — max denominator")
print("=" * 78)
for sk, label in [("small", "small"), ("medium", "medium"), ("large_v3", "large-v3")]:
    tf = os.path.join(BASE, f"master_metadata_whisper_{sk}_temp_sweep.csv")
    if not os.path.exists(tf):
        continue
    dt = load(tf)
    bk = sk.replace("-", "_")
    t0 = mean_dps(dt, f"whisper_{bk}_T0_0")
    t1 = mean_dps(dt, f"whisper_{bk}_T1_0")
    print(f"Whisper {label:<10} T0={t0:+.4f}  T1={t1:+.4f}  delta={t1-t0:+.4f}")

print("\n" + "=" * 78)
print("LONG-FORM (single utterance, no_prev_text) — max denominator")
print("=" * 78)
intv_cols = {"tiny": "whisper_tiny_no_prev_text", "small": "whisper_small_no_prev_text",
             "medium": "whisper_medium_no_prev_text", "large_v3": "whisper_large_v3_no_prev_text"}
base_cols = {"tiny": "whisper_tiny_text", "small": "whisper_small_text",
             "medium": "whisper_medium_text", "large_v3": "whisper_large_v3_text"}
for sk, label in [("tiny", "tiny"), ("small", "small"), ("medium", "medium"), ("large_v3", "large-v3")]:
    bf = os.path.join(BASE, f"master_metadata_whisper_{sk}.csv")
    inf = os.path.join(BASE, f"master_metadata_whisper_{sk}_no_prev_text.csv")
    if not os.path.exists(inf):
        continue
    mb = mean_dps(load(bf), base_cols[sk])
    mi = mean_dps(load(inf), intv_cols[sk])
    print(f"Whisper {label:<10} base={mb:+.4f}  no_prev={mi:+.4f}  delta={mi-mb:+.4f}")

print("\n" + "=" * 78)
print("LONG-FORM (concatenated ~80 chunks, condition toggle) — max denominator")
print("=" * 78)
import glob
LF = os.path.join(BASE, "longform")
for sk, label in [("large_v3", "large-v3")]:
    files = glob.glob(os.path.join(LF, f"longform_*_whisper_{sk}_toggle.csv"))
    df = pd.concat([pd.read_csv(f) for f in files])
    dps_true, dps_false = [], []
    for _, r in df.iterrows():
        ds = sim(r["pred_text"], r["ref_dialect"])
        ns = sim(r["pred_text"], r["ref_standard"])
        if max(ds, ns) < 0.1:
            continue
        (dps_true if str(r["condition"]).lower() == "true" else dps_false).append(ds - ns)
    mt, mf = np.mean(dps_true), np.mean(dps_false)
    print(f"Whisper {label}: DPS(True)={mt:+.4f}  DPS(False)={mf:+.4f}  "
          f"delta(False-True)={mf-mt:+.4f}")

print("\n" + "=" * 78)
print("BETTHUPFERL replication — max denominator")
print("=" * 78)
BET = "/Users/1uckyeom/Desktop/GermanDialect/results/betthubferl"
for name, f in [("tiny", "openai-whisper-tiny_zeroshot_detailed.tsv"),
                ("small", "openai-whisper-small_zeroshot_detailed.tsv"),
                ("medium", "openai-whisper-medium_zeroshot_detailed.tsv"),
                ("large-v3", "openai-whisper-large-v3_zeroshot_detailed.tsv")]:
    fp = os.path.join(BET, f)
    if not os.path.exists(fp):
        continue
    db = pd.read_csv(fp, sep="\t")
    ds = [sim(r["PRED"], r["GOLD_DIAL"]) for _, r in db.iterrows()]
    ns = [sim(r["PRED"], r["GOLD_DEU"]) for _, r in db.iterrows()]
    dps = [d - n for d, n in zip(ds, ns) if max(d, n) >= 0.1]
    print(f"Whisper {name:<10} DialectSim={np.mean(ds):.3f}  NormSim={np.mean(ns):.3f}  "
          f"DPS={np.mean(dps):+.3f}")
