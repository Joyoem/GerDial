#!/usr/bin/env python3
"""
2D scatter: DialectSim vs NormSim per utterance, with the DPS=0 diagonal.

Rebuttal evidence #1. The reviewer raised two concerns:

  (1) DPS = DialectSim - NormSim collapses three very different cases
      (1.0/1.0, 0.5/0.5, 0.0/0.0) all to 0, obscuring information.
  (2) The two scores may not be on a comparable scale, biasing DPS.

This figure answers both directly on the raw per-utterance data:

  * The full 2D information is retained --- DPS is only the *signed* distance
    of a point from the diagonal. The two components are always reported
    alongside DPS, so nothing is "obscured".
  * The points are NOT uniformly offset to one side of the diagonal (which is
    what a systematic scale bias would look like). They straddle the diagonal,
    and their position is structured by model family.

Runs on CPU from the existing inference CSVs. No new inference, no GPU.

Usage:
    python scatter_dps.py [--denom max|ref]

Dependencies: pandas, numpy, matplotlib, rapidfuzz (all already used by the repo).
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein

BASE = "/Users/1uckyeom/Desktop/GermanDialect/results/swissdial"
FIGDIR = "/Users/1uckyeom/Desktop/GermanDialect/figures"
os.makedirs(FIGDIR, exist_ok=True)

# Okabe-Ito colorblind-safe palette
FAMILY_COLOR = {"CTC": "#E69F00", "Whisper": "#56B4E9", "LLM-ASR": "#009E73"}

MODELS = [
    # name, csv filename, prediction column, family
    ("wav2vec2 300M",   "master_metadata_wav2vec_300m.csv",    "pred_text_wav2vec_300m", "CTC"),
    ("wav2vec2 1B",     "master_metadata_wav2vec_1b.csv",      "pred_text_wav2vec_1b",   "CTC"),
    ("Whisper tiny",     "master_metadata_whisper_tiny.csv",    "whisper_tiny_text",      "Whisper"),
    ("Whisper small",    "master_metadata_whisper_small.csv",   "whisper_small_text",     "Whisper"),
    ("Whisper medium",   "master_metadata_whisper_medium.csv",  "whisper_medium_text",    "Whisper"),
    ("Whisper large-v3", "master_metadata_whisper_large_v3.csv","whisper_large_v3_text",  "Whisper"),
    ("Qwen3-ASR 0.6B",   "master_metadata_qwen3_06b.csv",      "Qwen3-ASR-0.6B_text",    "LLM-ASR"),
    ("Qwen3-ASR 1.7B",   "master_metadata_qwen3_17b.csv",      "qwen3_17b_text",         "LLM-ASR"),
]

SHORT = {
    "wav2vec2 300M": "w2v 300M", "wav2vec2 1B": "w2v 1B",
    "Whisper tiny": "tiny", "Whisper small": "small",
    "Whisper medium": "medium", "Whisper large-v3": "large-v3",
    "Qwen3-ASR 0.6B": "Qwen 0.6B", "Qwen3-ASR 1.7B": "Qwen 1.7B",
}

# Paper Table 1 values (bootstrap means), hardcoded for cross-check.
TABLE1 = {
    "wav2vec2 300M":   (0.670, 0.567, +0.103),
    "wav2vec2 1B":     (0.673, 0.594, +0.079),
    "Whisper tiny":     (0.647, 0.608, +0.039),
    "Whisper small":    (0.699, 0.753, -0.055),
    "Whisper medium":   (0.712, 0.816, -0.104),
    "Whisper large-v3": (0.718, 0.834, -0.116),
    "Qwen3-ASR 0.6B":   (0.728, 0.727, +0.001),
    "Qwen3-ASR 1.7B":   (0.747, 0.764, -0.016),
}


def sim(hyp, ref, denom="max"):
    """Character-level normalized similarity. denom: 'max' (paper Eq 1-2) or 'ref' (old script)."""
    if pd.isna(hyp) or pd.isna(ref):
        return 0.0
    hyp, ref = str(hyp).strip(), str(ref).strip()
    if not hyp or not ref:
        return 0.0
    d = Levenshtein.distance(hyp, ref)
    den = max(len(hyp), len(ref)) if denom == "max" else len(ref)
    return max(0.0, 1.0 - d / den)


def per_utterance(df, pred_col, denom):
    """Return a DataFrame with per-utterance DialectSim, NormSim, DPS, collapsed."""
    ds, ns, dps, collapsed = [], [], [], []
    for _, r in df.iterrows():
        d = sim(r[pred_col], r["ref_swiss_german"], denom)
        n = sim(r[pred_col], r["ref_high_german"], denom)
        is_collapsed = max(d, n) < 0.1
        ds.append(d); ns.append(n)
        dps.append(None if is_collapsed else d - n)
        collapsed.append(is_collapsed)
    return pd.DataFrame({"DialectSim": ds, "NormSim": ns, "DPS": dps, "collapsed": collapsed})


def load_all(denom):
    out = {}
    for name, fname, col, _fam in MODELS:
        fp = os.path.join(BASE, fname)
        if not os.path.exists(fp):
            print(f"[skip] {fname} missing")
            continue
        out[name] = per_utterance(pd.read_csv(fp, encoding="utf-8-sig"), col, denom)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--denom", default="max", choices=["max", "ref"],
                    help="normalization denominator for the figure (default: paper's max)")
    args = ap.parse_args()

    # ── 1. Cross-check against Table 1 under BOTH denominators ──
    print("=" * 96)
    print("CROSS-CHECK: which denominator reproduces the paper's Table 1?")
    print("=" * 96)
    print(f"{'Model':<18}{'T1 DS':>7}{'T1 NS':>7}{'T1 DPS':>8} | "
          f"{'max DS':>7}{'max NS':>7}{'max DPS':>8} | "
          f"{'ref DS':>7}{'ref NS':>7}{'ref DPS':>8}")
    print("-" * 96)
    data_max = load_all("max")
    data_ref = load_all("ref")
    for name, _f, _c, _fam in MODELS:
        if name not in data_max:
            continue
        t1 = TABLE1[name]
        m = data_max[name]; r = data_ref[name]
        mm = (m["DialectSim"].mean(), m["NormSim"].mean(), m["DPS"].mean())
        rm = (r["DialectSim"].mean(), r["NormSim"].mean(), r["DPS"].mean())
        print(f"{name:<18}{t1[0]:>7.3f}{t1[1]:>7.3f}{t1[2]:>+8.3f} | "
              f"{mm[0]:>7.3f}{mm[1]:>7.3f}{mm[2]:>+8.3f} | "
              f"{rm[0]:>7.3f}{rm[1]:>7.3f}{rm[2]:>+8.3f}")
    print("\n(Whichever column tracks Table 1 most closely is the one actually used.)")

    # ── 2. Figures using the requested denominator ──
    data = data_max if args.denom == "max" else data_ref
    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    n_collapsed = 0
    for name, _f, _c, fam in MODELS:
        if name not in data:
            continue
        d = data[name]
        ok = d[~d["collapsed"]]
        n_collapsed += int(d["collapsed"].sum())
        color = FAMILY_COLOR[fam]
        ax.scatter(ok["NormSim"], ok["DialectSim"], s=4, alpha=0.10, color=color,
                   rasterized=True, label=fam if fam not in ax.get_legend_handles_labels()[1] else "")
        # per-model mean marker
        ax.scatter(ok["NormSim"].mean(), ok["DialectSim"].mean(), s=90,
                   marker="X", color=color, edgecolor="black", linewidth=0.8, zorder=5)
        ax.annotate(SHORT[name], (ok["NormSim"].mean(), ok["DialectSim"].mean()),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    # diagonal = DPS == 0
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], color="0.4", linestyle="--", linewidth=1.0, zorder=3)
    ax.text(0.98, 0.97, "DPS = 0 (no lean)", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="0.4")
    ax.text(0.02, 0.97, "above: dialect-leaning\n(DPS > 0)", transform=ax.transAxes,
            va="top", fontsize=8, color="0.2")
    ax.text(0.98, 0.03, "below: standard-leaning\n(DPS < 0)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="0.2")

    ax.set_xlabel("NormSim  (similarity to Standard German reference)")
    ax.set_ylabel("DialectSim  (similarity to dialect reference)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(markerscale=2, frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Per-utterance DialectSim vs NormSim (X = per-model mean)", fontsize=10)
    fig.tight_layout()
    out1 = os.path.join(FIGDIR, "figure_dps_2d_scatter.png")
    fig.savefig(out1, dpi=200)
    print(f"\nSaved scatter -> {out1}")
    print(f"(excluded {n_collapsed} collapsed utterances with max(DialectSim,NormSim) < 0.1)")

    # ── 3. Per-model DPS lollipop (architecture signature at a glance) ──
    order = [m[0] for m in MODELS if m[0] in data]
    fig2, ax2 = plt.subplots(figsize=(6.4, 3.6))
    ys = np.arange(len(order))[::-1]
    for y, name in zip(ys, order):
        fam = dict((m[0], m[3]) for m in MODELS)[name]
        dps = data[name]["DPS"].mean()
        ax2.plot([0, dps], [y, y], color=FAMILY_COLOR[fam], linewidth=1.5, zorder=2)
        ax2.scatter([dps], [y], s=70, color=FAMILY_COLOR[fam], edgecolor="black", linewidth=0.8, zorder=3)
    ax2.axvline(0, color="0.4", linewidth=1.0)
    ax2.set_yticks(ys); ax2.set_yticklabels([SHORT[n] for n in order], fontsize=8)
    ax2.set_xlabel("DPS (DialectSim - NormSim)")
    ax2.set_title("Architecture signature: DPS by model", fontsize=10)
    ax2.grid(axis="x", color="0.9", linewidth=0.5)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    fig2.tight_layout()
    out2 = os.path.join(FIGDIR, "figure_dps_per_model.png")
    fig2.savefig(out2, dpi=200)
    print(f"Saved lollipop -> {out2}")


if __name__ == "__main__":
    main()
