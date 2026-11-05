#!/usr/bin/env python3
"""
Concatenate same-dialect sentences into long-form audio for cross-chunk testing.
Uses scipy (handles int32/float WAVs that wave module rejects).

Output per dialect:
  data1.1/longform/{dialect}/audio.wav      — 16-bit PCM, 22050 Hz mono
  data1.1/longform/{dialect}/manifest.csv   — sentence_id, start_sec, end_sec, ref_dialect, ref_std

Usage:
  python scripts/concat_longform.py                  # all 8 dialects, 500 sentences each
  python scripts/concat_longform.py --n 200           # 200 per dialect
  python scripts/concat_longform.py --dialects zh,be  # specific dialects
"""
import argparse, os, sys
import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR   = PROJECT_ROOT / "data1.1"
METADATA    = PROJECT_ROOT / "master_metadata.csv"
OUT_ROOT    = AUDIO_DIR / "longform"

SILENCE_SEC = 0.5
DIALECTS    = ["ag","be","bs","gr","lu","sg","vs","zh"]


def concat_dialect(dialect: str, n: int):
    """Concatenate first N sentences of a dialect. scipy reads/writes, no wave module."""
    df = pd.read_csv(METADATA)
    df_d = df[df["dialect"].str.lower() == dialect.lower()].head(n).reset_index(drop=True)
    n_actual = len(df_d)
    if n_actual == 0:
        print(f"  [{dialect}] 0 sentences, skip")
        return

    out_dir = OUT_ROOT / dialect
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav_path = out_dir / "audio.wav"

    # ── Read all audio in one pass (each file ~440KB, 500 files = 220MB — fine) ──
    sr = None
    segments = []   # list of (float32 array) for each sentence + silence
    manifest_rows = []
    current_time = 0.0

    for _, row in df_d.iterrows():
        fname = os.path.basename(row["audio_path"])
        fpath = AUDIO_DIR / dialect / fname
        if not fpath.exists():
            sys.exit(f"MISSING: {fpath}")

        rate, data = wav.read(str(fpath))
        if sr is None:
            sr = rate
        else:
            assert rate == sr, f"Sample rate mismatch: {fpath} ({rate} vs {sr})"

        # Convert to mono float32 [-1, 1]
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Handle int32, int16, float — normalize to [-1, 1]
        if data.dtype == np.int32:
            data_f = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.int16:
            data_f = data.astype(np.float32) / 32768.0
        elif np.issubdtype(data.dtype, np.floating):
            data_f = data.astype(np.float32)
        else:
            data_f = data.astype(np.float32)
            mx = np.abs(data_f).max()
            if mx > 0:
                data_f /= mx

        duration = len(data_f) / sr
        start_t = current_time
        end_t = current_time + duration

        manifest_rows.append({
            "sentence_id": row["sentence_id"],
            "dialect": dialect,
            "start_sec": round(start_t, 3),
            "end_sec": round(end_t, 3),
            "ref_dialect": str(row["ref_swiss_german"]).strip(),
            "ref_standard": str(row["ref_high_german"]).strip(),
        })

        segments.append(data_f)
        # Silence gap
        silence = np.zeros(int(sr * SILENCE_SEC), dtype=np.float32)
        segments.append(silence)

        current_time = end_t + SILENCE_SEC

    # ── Concatenate & write as 16-bit PCM ──
    full = np.concatenate(segments)
    mx = np.abs(full).max()
    if mx > 0:
        full /= mx
    full_int16 = (full * 32767).astype(np.int16)
    wav.write(str(out_wav_path), sr, full_int16)

    # ── Save manifest ──
    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)

    dur_min = current_time / 60
    chunks = current_time / 30
    mem_mb = full_int16.nbytes / 1024 / 1024
    print(f"  [{dialect}] {n_actual} sentences → {dur_min:.1f} min, ~{chunks:.0f} chunks, "
          f"{mem_mb:.0f} MB → {out_wav_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="Sentences per dialect")
    p.add_argument("--dialects", type=str, default="all")
    args = p.parse_args()

    dialects = DIALECTS if args.dialects == "all" else [d.strip().lower() for d in args.dialects.split(",")]

    print(f"Concatenating {args.n} sentences × {len(dialects)} dialects\n")
    for d in dialects:
        concat_dialect(d, args.n)
    print(f"\n[DONE] Long-form audio ready in {OUT_ROOT}/")
    print("Next: python scripts/run_longform_toggle.py --model large-v3 --dialect zh")


if __name__ == "__main__":
    main()
