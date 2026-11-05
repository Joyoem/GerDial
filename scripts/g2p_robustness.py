#!/usr/bin/env python3
"""
g2p robustness: recompute DPS at the PHONEME level (instead of character level).

Rebuttal evidence #2. The reviewer argued the two scores may not be directly
comparable because dialect transcription has multiple valid spellings, so
character-level similarity to a *single* dialect reference may be systematically
underestimated. If that were true, the character-level result would be an
orthographic artifact.

This script re-measures similarity in phonetic space (grapheme-to-phoneme),
which is tolerant to spelling variation (two spellings that sound the same map
to the same phoneme string). If the tripartite architecture signature (CTC+,
Whisper monotonic negative, LLM-ASR ~0) survives, the DPS signal is NOT an
artifact of orthography or scale.

Runs on CPU. No inference, no GPU.

g2p backend (auto-detected, first available):
  1. `phonemizer` + espeak-ng  (highest quality German IPA; needs espeak-ng binary)
        brew install espeak-ng && pip install phonemizer
  2. `gruut`                   (pure Python, German IPA, no system binary)  <- used here
        pip install gruut gruut-ipa gruut-lang-de
  3. `epitran`                 (pure Python, weaker German)
        pip install epitran

Note: all three are STANDARD German g2p. Swiss dialect text is phonemized with a
standard-German sound inventory. This is fine for a *robustness* check (hypothesis
and both references go through the SAME g2p), but do not claim it models Swiss
phonology precisely.

Because gruut is slow (~136 ms/sentence) and the CSVs hold ~31k utterances each,
we sample `--sample` utterances (default 2000, fixed seed, SAME rows across all
models) and phonemize in parallel with `--workers` processes. A 2000-utterance
sample has mean SE ~0.006, far below the 0.02--0.1 DPS differences we test.

Usage:
    python g2p_robustness.py [--sample 2000] [--seed 42] [--workers 8]
"""
import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

BASE = "/Users/1uckyeom/Desktop/GermanDialect/results/swissdial"
OUT = "/Users/1uckyeom/Desktop/GermanDialect/results/g2p_robustness.csv"

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


# ── module-level g2p workers (picklable, called by the process pool) ──
def _gruut_one(text):
    from gruut import sentences
    parts = []
    for sent in sentences(text, lang="de"):
        for w in sent.words:
            if w.phonemes:
                parts.append("".join(w.phonemes))  # concat phonemes within word
    return " ".join(parts)                        # space between words


def _epitran_one(text):
    import epitran
    epi = epitran.Epitran("deu-Latn")
    return epi.transliterate(text)


def _espeak_batch(texts):
    from phonemizer import phonemize
    return phonemize(texts, language="de", backend="espeak", strip=True,
                     preserve_punctuation=False)


def pick_backend():
    """Return the name of the best available backend."""
    try:
        import phonemizer  # noqa: F401
        return "espeak"
    except Exception:
        pass
    try:
        import gruut  # noqa: F401
        return "gruut"
    except Exception:
        pass
    try:
        import epitran  # noqa: F401
        return "epitran"
    except Exception:
        return None


def clean_ipa(s):
    s = re.sub(r"[ˈˌ]", "", str(s))        # stress marks
    s = re.sub(r"\s+", " ", s).strip()     # normalize whitespace
    return s


def char_sim(hyp, ref):
    """Character-level similarity (paper Eq 1-2, max denominator)."""
    if pd.isna(hyp) or pd.isna(ref):
        return 0.0
    hyp, ref = str(hyp).strip(), str(ref).strip()
    if not hyp or not ref:
        return 0.0
    d = Levenshtein.distance(hyp, ref)
    return max(0.0, 1.0 - d / max(len(hyp), len(ref)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    backend = pick_backend()
    if backend is None:
        print("No g2p backend available. Install one of:")
        print("  pip install gruut gruut-ipa gruut-lang-de")
        print("  brew install espeak-ng && pip install phonemizer")
        print("  pip install epitran")
        raise SystemExit(1)
    print(f"g2p backend: {backend}   sample={args.sample}   workers={args.workers}\n")

    # ── 1. load all CSVs, sample the SAME row indices across models ──
    dfs, cols = {}, {}
    for name, fname, col, _fam in MODELS:
        fp = os.path.join(BASE, fname)
        if not os.path.exists(fp):
            print(f"[skip] {fname} missing")
            continue
        dfs[name] = pd.read_csv(fp, encoding="utf-8-sig")
        cols[name] = col

    n = min(len(df) for df in dfs.values())
    k = min(args.sample, n)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    print(f"Sampling {k} utterances (same rows across {len(dfs)} models).")

    # ── 2. collect every unique text we need to phonemize ──
    unique = set()
    for name, df in dfs.items():
        sub = df.iloc[idx]
        for _, r in sub.iterrows():
            for val in (r["ref_swiss_german"], r["ref_high_german"], r[cols[name]]):
                if pd.isna(val):
                    continue
                t = str(val).strip().lower()
                if t:
                    unique.add(t)
    unique = list(unique)
    print(f"Phonemizing {len(unique)} unique strings...")

    # ── 3. phonemize (parallel for slow backends, batch for espeak) ──
    if backend == "espeak":
        ph_list = [clean_ipa(x) for x in _espeak_batch(unique)]
    else:
        worker = _gruut_one if backend == "gruut" else _epitran_one
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            ph_list = list(ex.map(worker, unique, chunksize=32))
    ph_cache = {u: clean_ipa(p) for u, p in zip(unique, ph_list)}
    print("done.\n")

    def ph_sim(hyp, ref):
        if pd.isna(hyp) or pd.isna(ref):
            return 0.0
        hp = ph_cache.get(str(hyp).strip().lower(), "")
        rp = ph_cache.get(str(ref).strip().lower(), "")
        if not hp or not rp:
            return 0.0
        d = Levenshtein.distance(hp, rp)
        return max(0.0, 1.0 - d / max(len(hp), len(rp)))

    # ── 4. char-level vs phoneme-level, on the SAME sample ──
    print("=" * 92)
    print(f"{'Model':<18}{'DS_char':>8}{'NS_char':>8}{'DPS_char':>9} | "
          f"{'DS_ph':>8}{'NS_ph':>8}{'DPS_ph':>9}   signature?")
    print("-" * 92)

    rows = []
    for name, df in dfs.items():
        sub = df.iloc[idx]
        ds_c = [char_sim(r[cols[name]], r["ref_swiss_german"]) for _, r in sub.iterrows()]
        ns_c = [char_sim(r[cols[name]], r["ref_high_german"]) for _, r in sub.iterrows()]
        dps_c = [d - n if max(d, n) >= 0.1 else np.nan for d, n in zip(ds_c, ns_c)]

        ds_p = [ph_sim(r[cols[name]], r["ref_swiss_german"]) for _, r in sub.iterrows()]
        ns_p = [ph_sim(r[cols[name]], r["ref_high_german"]) for _, r in sub.iterrows()]
        dps_p = [d - n if max(d, n) >= 0.1 else np.nan for d, n in zip(ds_p, ns_p)]

        mc = (np.nanmean(ds_c), np.nanmean(ns_c), np.nanmean(dps_c))
        mp = (np.nanmean(ds_p), np.nanmean(ns_p), np.nanmean(dps_p))

        sig = ("dialect+" if mp[2] > 0.02 else "standard-" if mp[2] < -0.02 else "~0")
        print(f"{name:<18}{mc[0]:>8.3f}{mc[1]:>8.3f}{mc[2]:>+9.3f} | "
              f"{mp[0]:>8.3f}{mp[1]:>8.3f}{mp[2]:>+9.3f}   {sig}")

        rows.append({"model": name, "family": dict((m[0], m[3]) for m in MODELS)[name],
                     "DialectSim_char": mc[0], "NormSim_char": mc[1], "DPS_char": mc[2],
                     "DialectSim_ph": mp[0], "NormSim_ph": mp[1], "DPS_ph": mp[2]})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved per-model table -> {OUT}")

    # ── 5. verdict ──
    print("\n" + "=" * 92)
    print("VERDICT: does the architecture signature survive at the phoneme level?")
    print("=" * 92)
    fams = {"CTC": [], "Whisper": [], "LLM-ASR": []}
    for r in rows:
        fams[r["family"]].append(r["DPS_ph"])
    checks = [
        ("CTC positive",            all(d > 0 for d in fams["CTC"])),
        ("Whisper monotonic decr.", fams["Whisper"] == sorted(fams["Whisper"], reverse=True)),
        ("Whisper ends negative",   fams["Whisper"][-1] < 0),
        ("LLM-ASR near zero",       all(abs(d) < 0.02 for d in fams["LLM-ASR"])),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'CHANGE'}] {label:<24} Whisper={[round(x,3) for x in fams['Whisper']]}")
    print("\nInterpretation: if the first three PASS, the DPS signal is robust to")
    print("orthographic/spelling variation and is not an artifact of character-level scale.")


if __name__ == "__main__":
    main()
