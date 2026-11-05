#!/usr/bin/env python3
"""
Long-form causal test for condition_on_previous_text.

Short SwissDial sentences (3-7s) fit in a single 30s Whisper chunk → the
parameter has no effect. This script concatenates same-dialect sentences into
multi-minute audio that spans multiple chunks. The hypothesis:

  condition_on_previous_text=True  → chunk N's (standardized) output feeds
                                      into chunk N+1's decoder → DPS ↓
  condition_on_previous_text=False → each chunk is independent → DPS higher

Usage:
  python scripts/longform_causal_test.py --dialect zh --n 60 --model tiny --test
  python scripts/longform_causal_test.py --dialect zh --n 60 --model large-v3
"""

import argparse, os, sys, time, tempfile
import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR   = PROJECT_ROOT / "data1.1"
METADATA    = PROJECT_ROOT / "master_metadata.csv"
OUT_DIR     = PROJECT_ROOT / "results" / "swissdial"

SILENCE_SEC = 0.5          # gap between sentences
DIALECTS    = ["ag","be","bs","gr","lu","sg","vs","zh"]


def build_longform(dialect: str, n: int):
    """
    Concatenate first N sentences of a dialect into one long audio file.
    Returns: (wav_path, concat_dialect_ref, concat_standard_ref, sentence_count)
    """
    df = pd.read_csv(METADATA)
    df_d = df[df["dialect"].str.lower() == dialect.lower()].head(n).reset_index(drop=True)
    if len(df_d) < n:
        sys.exit(f"Only {len(df_d)} sentences for {dialect}, need {n}")

    audio_dir = AUDIO_DIR / dialect.lower()
    sr = None
    audio_segments = []
    ref_dialect_parts = []
    ref_standard_parts = []

    for _, row in df_d.iterrows():
        fname = os.path.basename(row["audio_path"])
        fpath = audio_dir / fname
        if not fpath.exists():
            sys.exit(f"Missing: {fpath}")

        rate, data = wav.read(str(fpath))
        if sr is None:
            sr = rate
        elif rate != sr:
            sys.exit(f"Sample rate mismatch: {fpath} ({rate} vs {sr})")

        # Convert to mono float32 normalized
        if data.ndim > 1:
            data = data.mean(axis=1)
        data_f = data.astype(np.float32) / max(np.abs(data).max(), 1.0)

        audio_segments.append(data_f)
        ref_dialect_parts.append(str(row["ref_swiss_german"]).strip())
        ref_standard_parts.append(str(row["ref_high_german"]).strip())

        # Insert silence
        silence = np.zeros(int(sr * SILENCE_SEC), dtype=np.float32)
        audio_segments.append(silence)

    # Concatenate everything
    full_audio = np.concatenate(audio_segments)
    # Normalize to [-1, 1] then convert to int16 for WAV
    full_audio = full_audio / max(np.abs(full_audio).max(), 1.0)
    full_int16 = (full_audio * 32767).astype(np.int16)

    out_name = f"longform_{dialect}_n{len(df_d)}.wav"
    out_path = AUDIO_DIR / out_name
    wav.write(str(out_path), sr, full_int16)

    # Concatenated references (with <|SILENCE|> markers matching audio gaps)
    ref_dia_concat = " <|gap|> ".join(ref_dialect_parts)
    ref_std_concat = " <|gap|> ".join(ref_standard_parts)

    duration_min = len(full_audio) / sr / 60
    chunks_est = len(full_audio) / sr / 30  # ~30s per chunk

    print(f"  Created: {out_name}")
    print(f"  Sentences: {len(df_d)}, Duration: {duration_min:.1f} min, ~{chunks_est:.0f} chunks")
    print(f"  Dialect ref: {len(ref_dia_concat)} chars, Standard ref: {len(ref_std_concat)} chars")

    return out_path, ref_dia_concat, ref_std_concat, len(df_d)


def run_toggle(audio_path: Path, ref_dia: str, ref_std: str,
               model_key: str, n_sentences: int):
    """
    Run condition_on_previous_text=True vs False on long-form audio.
    Compute DPS for both conditions.
    """
    from faster_whisper import WhisperModel
    from rapidfuzz.distance import Levenshtein

    ct = "int8_float16" if model_key == "large-v3" else "float16"
    print(f"\n  Loading faster-whisper '{model_key}' ({ct})...", flush=True)
    model = WhisperModel(model_key, device="cuda", compute_type=ct,
                         num_workers=2, cpu_threads=2)

    results = {}
    for cond_label, cond_val in [("True", True), ("False", False)]:
        print(f"  Transcribing (condition_on_previous_text={cond_val})...", flush=True)
        t0 = time.time()
        segs, _ = model.transcribe(
            str(audio_path),
            beam_size=5, best_of=5,
            language="de",
            condition_on_previous_text=cond_val,
            vad_filter=False,  # already clean with silence gaps
        )
        full_text = " ".join(s.text.strip() for s in segs)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.0f}s, output: {len(full_text)} chars", flush=True)

        # Compute DPS on full concatenated output vs concatenated references
        ds = 1.0 - Levenshtein.distance(full_text, ref_dia) / max(len(ref_dia), 1)
        ns = 1.0 - Levenshtein.distance(full_text, ref_std) / max(len(ref_std), 1)
        dps = ds - ns if max(ds, ns) >= 0.1 else None

        results[cond_label] = {
            "text_len": len(full_text),
            "DialectSim": ds,
            "NormSim": ns,
            "DPS": dps,
        }

    return results


def main():
    p = argparse.ArgumentParser(description="Long-form causal toggle test")
    p.add_argument("--dialect", default="zh", help="Dialect code (zh, be, etc.)")
    p.add_argument("--n", type=int, default=60, help="Sentences to concatenate")
    p.add_argument("--model", default="tiny", choices=["tiny","small","medium","large-v3"])
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    if args.test:
        args.n = min(args.n, 20)

    print("=" * 65)
    print(f"LONG-FORM CAUSAL TEST")
    print(f"  Dialect: {args.dialect}, Sentences: {args.n}, Model: {args.model}")
    print(f"  Hypothesis: DPS(True) < DPS(False)")
    print("=" * 65)

    # Build long audio
    audio_path, ref_dia, ref_std, n_actual = build_longform(args.dialect, args.n)

    # Run toggle
    results = run_toggle(audio_path, ref_dia, ref_std, args.model, n_actual)

    # Print results
    print(f"\n{'='*50}")
    print(f"{'Condition':<12} {'DialectSim':>10} {'NormSim':>10} {'DPS':>10}")
    print(f"{'-'*50}")
    for cond in ["True", "False"]:
        r = results[cond]
        dps_str = f"{r['DPS']:+.4f}" if r['DPS'] is not None else "N/A"
        print(f"prev_text={cond:<5} {r['DialectSim']:>10.4f} {r['NormSim']:>10.4f} {dps_str:>10}")

    dps_t = results["True"]["DPS"]
    dps_f = results["False"]["DPS"]
    if dps_t is not None and dps_f is not None:
        delta = dps_f - dps_t
        print(f"{'-'*50}")
        print(f"Δ (False − True): {delta:+.4f}")
        if delta > 0.005:
            print("✓ HYPOTHESIS CONFIRMED: cross-chunk feedback causes more standardization")
        elif abs(delta) < 0.005:
            print("— No significant difference: cross-chunk feedback not the driver")
        else:
            print("✗ Opposite direction: False more standardized than True")

    print(f"\nAudio saved: {audio_path}")


if __name__ == "__main__":
    main()
