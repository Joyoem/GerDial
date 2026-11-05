#!/usr/bin/env python3
"""
Long-form causal toggle: condition_on_previous_text=True vs False.

Runs on pre-concatenated long-form audio (from concat_longform.py).
Matches Whisper segments back to original sentences using time overlap,
so per-sentence DPS can be compared between conditions.

Usage:
  python scripts/run_longform_toggle.py --model tiny --dialect zh
  python scripts/run_longform_toggle.py --model large-v3 --dialect zh
  python scripts/run_longform_toggle.py --model tiny --dialect all   # sweep
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from rapidfuzz.distance import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR   = PROJECT_ROOT / "data1.1"
LONGFORM_DIR = AUDIO_DIR / "longform"
OUT_DIR     = PROJECT_ROOT / "results" / "swissdial"

COMPUTE_TYPE = {
    "tiny": "float16", "small": "float16",
    "medium": "float16", "large-v3": "int8_float16",
}


def ncer(hyp, ref):
    if pd.isna(hyp) or pd.isna(ref): return 0.0
    hyp, ref = str(hyp).strip(), str(ref).strip()
    if not ref or not hyp: return 0.0
    return max(0.0, 1.0 - Levenshtein.distance(hyp, ref) / len(ref))


def align_segments_to_sentences(segments, manifest_df):
    """
    Match Whisper segments to original sentences by time overlap.
    Each sentence gets the concatenated text of all segments that overlap with it.
    """
    preds = []
    for _, sent in manifest_df.iterrows():
        s_start, s_end = sent["start_sec"], sent["end_sec"]
        overlapping_texts = []
        for seg in segments:
            # Check overlap
            if seg.end > s_start and seg.start < s_end:
                overlapping_texts.append(seg.text.strip())
        preds.append(" ".join(overlapping_texts))
    return preds


def run_single(model_key: str, dialect: str):
    from faster_whisper import WhisperModel

    wav_path = LONGFORM_DIR / dialect / "audio.wav"
    manifest_path = LONGFORM_DIR / dialect / "manifest.csv"
    if not wav_path.exists():
        print(f"  [{dialect}] Audio not found: {wav_path}"); return
    if not manifest_path.exists():
        print(f"  [{dialect}] Manifest not found: {manifest_path}"); return

    manifest = pd.read_csv(manifest_path)
    ct = COMPUTE_TYPE[model_key]
    base = model_key.replace("-", "_")
    out_csv = OUT_DIR / f"longform_{dialect}_whisper_{base}_toggle.csv"

    # Resume
    done_true = set()
    done_false = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done_true  = set(existing[existing["condition"]=="True"]["sentence_id"].astype(str))
        done_false = set(existing[existing["condition"]=="False"]["sentence_id"].astype(str))
        if done_true and done_false:
            print(f"  [{dialect}] Already complete ({len(done_true)} sentences)")
            return out_csv
        print(f"  [{dialect}] Resume: True={len(done_true)}, False={len(done_false)} done")

    print(f"  [{dialect}] Loading faster-whisper '{model_key}' ({ct})...", flush=True)
    model = WhisperModel(model_key, device="cuda", compute_type=ct,
                         num_workers=2, cpu_threads=2)

    all_rows = []

    # ── Collect existing rows ──
    if out_csv.exists():
        all_rows = pd.read_csv(out_csv).to_dict("records")

    for cond_val, cond_label in [(True, "True"), (False, "False")]:
        done_set = done_true if cond_val else done_false
        if len(done_set) >= len(manifest):
            continue  # all done for this condition

        print(f"    condition_on_previous_text={cond_label}...", flush=True)
        t0 = time.time()
        segs, _ = model.transcribe(
            str(wav_path),
            beam_size=5, best_of=5,
            language="de",
            condition_on_previous_text=cond_val,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        segments_list = list(segs)
        elapsed = time.time() - t0
        print(f"    {len(segments_list)} segments in {elapsed:.0f}s", flush=True)

        # Align to sentences
        preds = align_segments_to_sentences(segments_list, manifest)

        for i, (_, sent) in enumerate(manifest.iterrows()):
            sid = str(sent["sentence_id"])
            if sid in done_set:
                continue
            ds = ncer(preds[i], sent["ref_dialect"])
            ns = ncer(preds[i], sent["ref_standard"])
            dps = ds - ns if max(ds, ns) >= 0.1 else None
            all_rows.append({
                "sentence_id": sid,
                "dialect": dialect,
                "condition": cond_label,
                "pred_text": preds[i],
                "ref_dialect": sent["ref_dialect"],
                "ref_standard": sent["ref_standard"],
                "DialectSim": round(ds, 6),
                "NormSim": round(ns, 6),
                "DPS": round(dps, 6) if dps is not None else None,
            })

    # Save
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    return out_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["tiny","small","medium","large-v3"])
    p.add_argument("--dialect", default="zh")
    args = p.parse_args()

    dialects = [d.strip().lower() for d in args.dialect.split(",")] if args.dialect != "all" else \
               [d.name for d in LONGFORM_DIR.iterdir() if d.is_dir()]

    print(f"Long-form toggle: {args.model} × {len(dialects)} dialects")
    print(f"Hypothesis: DPS(False) > DPS(True)\n")

    results = {}
    for d in dialects:
        out = run_single(args.model, d)
        if out is None:
            continue

        # Quick summary
        df = pd.read_csv(out)
        for cond in ["True", "False"]:
            dps_vals = df[(df["condition"]==cond)]["DPS"].dropna()
            if len(dps_vals) > 0:
                results[(d, cond)] = np.mean(dps_vals)

    # Summary
    if results:
        print(f"\n{'='*55}")
        print(f"{'Dialect':<10} {'DPS(True)':>12} {'DPS(False)':>12} {'Δ':>10}")
        print(f"{'-'*55}")
        for d in dialects:
            t = results.get((d, "True"))
            f = results.get((d, "False"))
            if t is not None and f is not None:
                delta = f - t
                verdict = "✓ confirmed" if delta > 0.003 else ("✗ reversed" if delta < -0.003 else "— no effect")
                print(f"{d:<10} {t:>+12.4f} {f:>+12.4f} {delta:>+10.4f}  {verdict}")


if __name__ == "__main__":
    main()
