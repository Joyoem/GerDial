#!/usr/bin/env python3
"""
Causal toggle experiment: Whisper inference with condition_on_previous_text=False.

This isolates the decoder's self-feedback loop as a causal factor in dialect
standardization. The hypothesis: disabling previous-text conditioning forces
each chunk to rely more on acoustic evidence → DPS should rise (less negative).

Usage:
    # Quick test (tiny, first 200 sentences)
    python scripts/whisper_causal_toggle.py --model tiny --test

    # Full run
    python scripts/whisper_causal_toggle.py --model tiny
    python scripts/whisper_causal_toggle.py --model small
    python scripts/whisper_causal_toggle.py --model medium
    python scripts/whisper_causal_toggle.py --model large-v3

Existing results (condition_on_previous_text=True, the default) are already in
results/swissdial/master_metadata_whisper_*.csv.

Output: results/swissdial/master_metadata_whisper_{model}_no_prev_text.csv
"""

import argparse
import os
import sys
import pandas as pd
import whisper
import torch
import gc
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "master_metadata.csv"
AUDIO_DIR = PROJECT_ROOT / "data1.1"
RESULTS_DIR = PROJECT_ROOT / "results" / "swissdial"

MODEL_SIZES = {
    "tiny": "tiny",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",  # note: openai-whisper uses "large-v3", not "large_v3"
}


def resolve_audio_path(rel_path: str) -> str:
    """
    Map metadata paths (data1/ag/ch_ag_0000.wav) to data1.1/ag/ch_ag_0000.wav.
    """
    filename = os.path.basename(rel_path)
    dialect = rel_path.split("/")[1]  # data1/{dialect}/file.wav
    resolved = AUDIO_DIR / dialect / filename
    return str(resolved)


def load_metadata(test_mode: bool = False) -> pd.DataFrame:
    """Load master metadata and resolve audio paths."""
    df = pd.read_csv(METADATA_PATH)
    df["resolved_audio"] = df["audio_path"].apply(resolve_audio_path)

    # Verify first file exists
    first = df["resolved_audio"].iloc[0]
    if not os.path.exists(first):
        print(f"[FATAL] Audio not found: {first}")
        print(f"Expected audio under: {AUDIO_DIR}/")
        sys.exit(1)

    if test_mode:
        df = df.head(200)
        print(f"[TEST MODE] Using first {len(df)} sentences")

    return df


def get_output_path(model_key: str) -> Path:
    return RESULTS_DIR / f"master_metadata_whisper_{model_key.replace('-', '_')}_no_prev_text.csv"


def run_inference(model_key: str, test_mode: bool = False):
    model_name = MODEL_SIZES[model_key]
    output_path = get_output_path(model_key)

    # --- Load data ---
    df = load_metadata(test_mode)

    # --- Resume ---
    completed = set()
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            completed = set(existing["sentence_id"].astype(str))
            if len(completed) > 0:
                print(f"[RESUME] {len(completed)} sentences already done, skipping")
        except Exception:
            pass

    df_todo = df[~df["sentence_id"].astype(str).isin(completed)].copy()
    if len(df_todo) == 0:
        print("[DONE] All sentences already processed!")
        return

    print(f"Sentences to process: {len(df_todo)} / {len(df)}")

    # --- Load model ---
    print(f"\nLoading Whisper {model_name}...")
    torch.cuda.empty_cache()
    gc.collect()

    model = whisper.load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Model loaded on {model.device}")

    # --- Run inference ---
    col_name = f"whisper_{model_key.replace('-', '_')}_no_prev_text"

    # Write header if new file
    if not output_path.exists():
        pd.DataFrame(columns=[
            "sentence_id", "dialect", "audio_path",
            "ref_high_german", "ref_swiss_german", col_name
        ]).to_csv(output_path, index=False)

    for _, row in tqdm(df_todo.iterrows(), total=len(df_todo),
                       desc=f"Whisper {model_key} (no_prev_text)"):
        audio_file = row["resolved_audio"]

        try:
            result = model.transcribe(
                audio_file,
                condition_on_previous_text=False,  # ← THE KEY INTERVENTION
                language="de",
                verbose=False,
            )
            pred_text = result["text"].strip()
        except Exception as e:
            pred_text = f"ERROR: {str(e)[:80]}"

        record = {
            "sentence_id": row["sentence_id"],
            "dialect": row["dialect"],
            "audio_path": row["audio_path"],
            "ref_high_german": row["ref_high_german"],
            "ref_swiss_german": row["ref_swiss_german"],
            col_name: pred_text,
        }

        pd.DataFrame([record]).to_csv(
            output_path, mode="a", index=False, header=False
        )

    print(f"\n[DONE] Results saved to: {output_path}")
    print(f"Compare with: {RESULTS_DIR / f'master_metadata_whisper_{model_key.replace('-', '_')}.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Whisper causal toggle: condition_on_previous_text=False"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["tiny", "small", "medium", "large-v3"],
        help="Whisper model size"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test: first 200 sentences only"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Causal Toggle Experiment")
    print(f"  Model: Whisper {args.model}")
    print(f"  Intervention: condition_on_previous_text = False")
    print(f"  Baseline: existing results (condition_on_previous_text = True)")
    print(f"  Hypothesis: DPS(False) > DPS(True) — less standardization")
    print("=" * 60)

    run_inference(args.model, test_mode=args.test)


if __name__ == "__main__":
    main()
