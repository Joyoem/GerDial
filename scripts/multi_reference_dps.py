#!/usr/bin/env python3
"""
Multi-reference robustness for DPS.

Answers the reviewer's "multiple possible outputs" concern + the author's
"what if our two references are coincidental" worry, using the fact that
SwissDial's sentences are parallel across 8 dialect regions (so each sentence
has 8 dialect references + 1 standard reference, not just 2).

Three sub-experiments (all character-level, CPU-only, no inference):

  Exp A (multi-reference):  DPS computed with max-over-8-dialect-references
      (PolyWER-style) vs the single correct-dialect reference. If the two agree
      and the architecture signature survives, then the result does NOT depend
      on any single spelling -> "single reference undercounts dialect sim" is
      refuted.

  Exp B (wrong-dialect control):  for a hypothesis spoken in dialect X, compare
      DialectSim to X's own reference vs the mean of the OTHER 7 dialects'
      references. If correct > wrong, DPS tracks real dialect identity, not
      "any non-standard text" (a coincidence).

  Exp C (permutation null):  shuffle the dialect labels and recompute the
      correct-vs-wrong gap. The observed gap should be far outside the null.

Usage:
    python multi_reference_dps.py
"""
import os
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

BASE = "/Users/1uckyeom/Desktop/GermanDialect/results/swissdial"
OUT = "/Users/1uckyeom/Desktop/GermanDialect/results/multi_reference_dps.csv"

MODELS = [
    ("wav2vec2 300M",   "master_metadata_wav2vec_300m.csv",    "pred_text_wav2vec_300m", "CTC"),
    ("wav2vec2 1B",     "master_metadata_wav2vec_1b.csv",      "pred_text_wav2vec_1b",   "CTC"),
    ("Whisper tiny",     "master_metadata_whisper_tiny.csv",    "whisper_tiny_text",      "Whisper"),
    ("Whisper small",    "master_metadata_whisper_small.csv",   "whisper_small_text",     "Whisper"),
    ("Whisper medium",   "master_metadata_whisper_medium.csv",  "whisper_medium_text",    "Whisper"),
    ("Whisper large-v3", "master_metadata_whisper_large_v3.csv","whisper_large_v3_text",  "Whisper"),
    ("Qwen3-ASR 0.6B",   "master_metadata_qwen3_06b.csv",      "Qwen3-ASR-0.6B_text",    "LLM-ASR"),
    ("Qwen3-ASR 1.7B",   "master_metadata_qwen3_17b.csv",      "qwen3_17b_text",         "LLM-ASR"),
]

ALL_DIALECTS = ["AG", "BE", "BS", "GR", "LU", "SG", "VS", "ZH"]


def sim(hyp, ref):
    if pd.isna(hyp) or pd.isna(ref):
        return 0.0
    hyp, ref = str(hyp).strip(), str(ref).strip()
    if not hyp or not ref:
        return 0.0
    d = Levenshtein.distance(hyp, ref)
    return max(0.0, 1.0 - d / max(len(hyp), len(ref)))


def build_refs():
    """sentence_id -> {'std': ref_high, 'd': {dialect: ref_swiss}} for fully-parallel sentences."""
    df = pd.read_csv(os.path.join(BASE, "master_metadata_wav2vec_300m.csv"), encoding="utf-8-sig")
    refs = {}
    for sid, grp in df.groupby("sentence_id"):
        dmap = dict(zip(grp["dialect"], grp["ref_swiss_german"]))
        std = grp["ref_high_german"].iloc[0]
        if set(dmap.keys()) == set(ALL_DIALECTS):
            refs[sid] = {"std": std, "d": dmap}
    return refs


def load_model(fname, col):
    df = pd.read_csv(os.path.join(BASE, fname), encoding="utf-8-sig")
    # qwen3_06b stores sentence_id as "0_ag" instead of 0; normalize to int for all CSVs
    df["sentence_id"] = df["sentence_id"].apply(lambda s: int(str(s).split("_")[0]))
    df = df.drop_duplicates(subset=["sentence_id", "dialect"], keep="first")
    return df


def evaluate(model_df, col, refs):
    """Return arrays: dps_correct, dps_max, dps_wrong, gap (correct_ds - wrong_ds)."""
    dps_correct, dps_max, dps_wrong, gaps = [], [], [], []
    for _, r in model_df.iterrows():
        sid, dia = r["sentence_id"], r["dialect"]
        if sid not in refs:  # only fully-parallel sentences
            continue
        hyp = r[col]
        dmap = refs[sid]["d"]
        ns = sim(hyp, refs[sid]["std"])
        ds_correct = sim(hyp, dmap[dia])
        ds_all = [sim(hyp, dmap[d]) for d in ALL_DIALECTS]
        ds_max = max(ds_all)
        ds_wrong = np.mean([ds_all[i] for i, d in enumerate(ALL_DIALECTS) if d != dia])
        if max(ds_correct, ns) < 0.1:  # collapse filter, consistent with paper
            continue
        dps_correct.append(ds_correct - ns)
        dps_max.append(ds_max - ns)
        dps_wrong.append(ds_wrong - ns)
        gaps.append(ds_correct - ds_wrong)
    return (np.array(dps_correct), np.array(dps_max), np.array(dps_wrong), np.array(gaps))


def main():
    refs = build_refs()
    print(f"Fully-parallel sentences (8 dialect references each): {len(refs)}")

    print("\n" + "=" * 100)
    print("Exp A (multi-ref) + Exp B (wrong-dialect control), char level")
    print("=" * 100)
    print(f"{'Model':<18}{'DPS_correct':>12}{'DPS_max(8ref)':>14}{'DPS_wrong':>12}"
          f"{'gap(c-w)':>10}   dialect-specific?")
    print("-" * 100)

    rows = []
    for name, fname, col, fam in MODELS:
        df = load_model(fname, col)
        c, m, w, g = evaluate(df, col, refs)
        print(f"{name:<18}{c.mean():>+12.3f}{m.mean():>+14.3f}{w.mean():>+12.3f}"
              f"{g.mean():>+10.3f}   {'YES' if g.mean() > 0 else 'no'}")
        rows.append({"model": name, "family": fam,
                     "DPS_correct": c.mean(), "DPS_max_8ref": m.mean(),
                     "DPS_wrong": w.mean(), "gap_correct_wrong": g.mean()})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")

    # ── Exp C: permutation null on the dialect-specificity gap ──
    print("\n" + "=" * 100)
    print("Exp C: permutation null test (is correct-vs-wrong gap a coincidence?)")
    print("=" * 100)
    rng = np.random.default_rng(0)
    for name, fname, col, _fam in MODELS:
        df = load_model(fname, col)
        # observed gap
        _, _, _, g_obs = evaluate(df, col, refs)
        obs = g_obs.mean()
        # null: shuffle dialect labels, recompute gap (correct label now random)
        null = []
        shuffled = df.copy()
        for _ in range(50):
            shuffled["dialect"] = rng.permutation(shuffled["dialect"].values)
            _, _, _, g_n = evaluate(shuffled, col, refs)
            null.append(g_n.mean())
        null = np.array(null)
        z = (obs - null.mean()) / (null.std() + 1e-12)
        p = (np.sum(null >= obs) + 1) / (len(null) + 1)
        print(f"{name:<18} obs gap={obs:+.4f}  null mean={null.mean():+.4f}±{null.std():.4f}"
              f"  z={z:>6.1f}  p<{p:.3f}")

    print("\nInterpretation:")
    print("  A: DPS_max ~= DPS_correct  -> result independent of any single spelling.")
    print("  B: gap(correct-wrong) > 0   -> DPS tracks the CORRECT dialect, not just 'non-standard'.")
    print("  C: z >> 2, p < 0.05          -> the dialect-specificity is not a coincidence.")


if __name__ == "__main__":
    main()
