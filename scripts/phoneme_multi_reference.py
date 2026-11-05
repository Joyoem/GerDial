#!/usr/bin/env python3
"""
Phoneme-level multi-reference DPS (the cross-product of the two robustness axes).

Combines:
  * the reference axis  (single vs max-over-8 vs wrong-dialect), and
  * the representation axis (orthographic vs phoneme-level via g2p).

Confirms the multi-reference + dialect-specificity results hold when similarity
is measured in phonetic space (tolerant to spelling variation), not just
character space.

Uses the same 2529 fully-parallel SwissDial sentences. Samples `--sample` of them
(default 400) because gruut phonemization is slow (~0.5 s/sentence); 400
parallel sentences = 3200 hypotheses/model, plenty to resolve the 0.06--0.09
dialect-specificity gap.

Usage:
    python phoneme_multi_reference.py [--sample 400] [--workers 8]
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

BASE = "/Users/1uckyeom/Desktop/GermanDialect/results/swissdial"
OUT = "/Users/1uckyeom/Desktop/GermanDialect/results/phoneme_multi_reference.csv"

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


def _gruut_one(text):
    from gruut import sentences
    parts = []
    for sent in sentences(text, lang="de"):
        for w in sent.words:
            if w.phonemes:
                parts.append("".join(w.phonemes))
    return " ".join(parts)


def _epitran_one(text):
    import epitran
    return epitran.Epitran("deu-Latn").transliterate(text)


def pick_backend():
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


def build_refs():
    df = pd.read_csv(os.path.join(BASE, "master_metadata_wav2vec_300m.csv"), encoding="utf-8-sig")
    refs = {}
    for sid, grp in df.groupby("sentence_id"):
        dmap = dict(zip(grp["dialect"], grp["ref_swiss_german"]))
        std = grp["ref_high_german"].iloc[0]
        if set(dmap.keys()) == set(ALL_DIALECTS):
            refs[int(sid)] = {"std": std, "d": dmap}
    return refs


def load_model(fname):
    df = pd.read_csv(os.path.join(BASE, fname), encoding="utf-8-sig")
    df["sentence_id"] = df["sentence_id"].apply(lambda s: int(str(s).split("_")[0]))
    df = df.drop_duplicates(subset=["sentence_id", "dialect"], keep="first")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    backend = pick_backend()
    if backend is None:
        print("No g2p backend. Install: pip install gruut gruut-ipa gruut-lang-de")
        raise SystemExit(1)
    print(f"backend={backend}  sample={args.sample}  workers={args.workers}")

    refs = build_refs()
    all_sids = list(refs.keys())
    rng = np.random.default_rng(42)
    sids = set(rng.choice(all_sids, size=min(args.sample, len(all_sids)), replace=False))
    print(f"Sampled {len(sids)} parallel sentences.")

    # collect unique strings
    unique = set()
    model_dfs = {}
    for name, fname, col, _fam in MODELS:
        df = load_model(fname).set_index(["sentence_id", "dialect"])
        model_dfs[name] = df
        sub = df[df.index.get_level_values("sentence_id").isin(sids)]
        for _, r in sub.iterrows():
            for val in (r["ref_swiss_german"], r["ref_high_german"], r[col]):
                if pd.isna(val):
                    continue
                t = str(val).strip().lower()
                if t:
                    unique.add(t)
    # also add all references for sampled sentences
    for sid in sids:
        unique.add(refs[sid]["std"].strip().lower())
        for d in ALL_DIALECTS:
            unique.add(refs[sid]["d"][d].strip().lower())
    unique = list(unique)
    print(f"Phonemizing {len(unique)} unique strings...")

    if backend == "espeak":
        from phonemizer import phonemize
        ph_list = [re.sub(r"[ˈˌ]", "", x) for x in
                   phonemize(unique, language="de", backend="espeak", strip=True,
                             preserve_punctuation=False)]
    else:
        worker = _gruut_one if backend == "gruut" else _epitran_one
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            ph_list = list(ex.map(worker, unique, chunksize=32))

    ph_cache = {u: re.sub(r"[ˈˌ]\s*", "", p).strip() for u, p in zip(unique, ph_list)}
    print("done.\n")

    def sim_ph(hyp, ref):
        hp = ph_cache.get(str(hyp).strip().lower(), "")
        rp = ph_cache.get(str(ref).strip().lower(), "")
        if not hp or not rp:
            return 0.0
        return max(0.0, 1.0 - Levenshtein.distance(hp, rp) / max(len(hp), len(rp)))

    print("=" * 96)
    print(f"{'Model':<18}{'DPS_correct':>12}{'DPS_max(8ref)':>14}{'DPS_wrong':>12}{'gap(c-w)':>10}")
    print("-" * 96)
    rows = []
    for name, fname, col, fam in MODELS:
        df = model_dfs[name]
        sub = df[df.index.get_level_values("sentence_id").isin(sids)]
        dps_c, dps_m, dps_w, gaps = [], [], [], []
        for (sid, dia), r in sub.iterrows():
            dmap = refs[sid]["d"]
            ns = sim_ph(r[col], refs[sid]["std"])
            ds_correct = sim_ph(r[col], dmap[dia])
            ds_all = [sim_ph(r[col], dmap[d]) for d in ALL_DIALECTS]
            ds_max = max(ds_all)
            ds_wrong = np.mean([ds_all[i] for i, d in enumerate(ALL_DIALECTS) if d != dia])
            if max(ds_correct, ns) < 0.1:
                continue
            dps_c.append(ds_correct - ns)
            dps_m.append(ds_max - ns)
            dps_w.append(ds_wrong - ns)
            gaps.append(ds_correct - ds_wrong)
        print(f"{name:<18}{np.mean(dps_c):>+12.3f}{np.mean(dps_m):>+14.3f}"
              f"{np.mean(dps_w):>+12.3f}{np.mean(gaps):>+10.3f}")
        rows.append({"model": name, "family": fam,
                     "DPS_correct_ph": np.mean(dps_c), "DPS_max_8ref_ph": np.mean(dps_m),
                     "DPS_wrong_ph": np.mean(dps_w), "gap_correct_wrong_ph": np.mean(gaps)})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
