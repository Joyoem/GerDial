#!/usr/bin/env python3
"""
Run Qwen3-ASR (0.6B and 1.7B) inference on the Betthupferl dataset.

Usage:
    python scripts/run_qwen3_asr_betthupferl.py --model 0.6B
    python scripts/run_qwen3_asr_betthupferl.py --model 1.7B

Audio must be present at: betthupferl/data_processed/audio/
Output format matches existing Betthupferl prediction TSVs (columns: INDEX, SPEAKER, DIALECT,
TRANSCRIPTION_DIALECT, TRANSCRIPTION_GERMAN, TRANSCRIPTION_MODEL).
"""

import argparse
import os
import sys
import pandas as pd
import torch
import gc
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BETTHUPFERL_ROOT = PROJECT_ROOT / "betthupferl"
TRANSCRIPT_DIR = BETTHUPFERL_ROOT / "data_processed" / "transcriptions"
AUDIO_DIR = BETTHUPFERL_ROOT / "data_processed" / "audio"
PRED_DIR = BETTHUPFERL_ROOT / "predictions"

MODEL_MAP = {
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
}

OUTPUT_DIR_TEMPLATE = "qwen3-asr-{model_key}_zeroshot"


def check_audio_available():
    """Verify audio files exist before starting inference."""
    if not AUDIO_DIR.exists():
        return False, f"Audio directory not found: {AUDIO_DIR}"
    wavs = list(AUDIO_DIR.glob("*.wav"))
    if len(wavs) == 0:
        return False, f"No .wav files found in {AUDIO_DIR}"
    return True, f"Found {len(wavs)} .wav files"


def load_transcriptions():
    """Load all Betthupferl transcription TSV files into one DataFrame."""
    records = []
    tsv_files = sorted(TRANSCRIPT_DIR.glob("*.tsv"))

    for tsv_path in tsv_files:
        story = tsv_path.stem  # e.g., "as-gloine-freilein-trudi-traudi-da-willi_oberpfalz"
        df = pd.read_csv(tsv_path, sep="\t")
        df["STORY"] = story
        records.append(df)

    full = pd.concat(records, ignore_index=True)
    # Filter out hochdeutsch control sentences (not dialect)
    full = full[~full["DIALECT"].str.startswith("hochdeutsch")]

    print(f"Loaded {len(full)} sentences from {len(tsv_files)} transcript files "
          f"(excl. hochdeutsch controls)")
    return full


def setup_output_dir(model_key: str) -> Path:
    """Create output directory mirroring existing prediction structure."""
    out_dir = PRED_DIR / OUTPUT_DIR_TEMPLATE.format(model_key=model_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def transcribe_with_qwen3_asr(model, audio_path: str) -> str:
    """
    Run inference on a single audio file.
    Assumes Qwen3-ASR model object with .transcribe() method.
    """
    try:
        result = model.transcribe(audio_path, language="de")
        return result["text"].strip() if isinstance(result, dict) else str(result).strip()
    except Exception as e:
        return f"ERROR: {str(e)[:100]}"


def run_inference(model_key: str, resume: bool = True):
    """Main inference loop."""
    # --- Validate ---
    ok, msg = check_audio_available()
    if not ok:
        print(f"[FATAL] {msg}")
        print("\nAudio files must be obtained from Bayerischer Rundfunk.")
        print("Contact Gabriele Wenger-Glemser (co-author, BR) for access.")
        print("Once obtained, place .wav files in:", AUDIO_DIR)
        sys.exit(1)
    print(f"[OK] {msg}")

    if model_key not in MODEL_MAP:
        print(f"[FATAL] Unknown model key: {model_key}. Choose from: {list(MODEL_MAP.keys())}")
        sys.exit(1)

    model_name = MODEL_MAP[model_key]
    out_dir = setup_output_dir(model_key)

    # --- Load data ---
    df = load_transcriptions()

    # --- Check resume ---
    completed = set()
    if resume:
        for tsv in out_dir.glob("*.tsv"):
            completed.add(tsv.stem)
        print(f"Resume: {len(completed)} stories already completed")

    # --- Group by story for per-story output TSV ---
    for story, group in tqdm(
        df.groupby("STORY"),
        desc=f"Qwen3-ASR {model_key}",
    ):
        if story in completed:
            continue

        output_rows = []
        for _, row in group.iterrows():
            idx = row["INDEX"]
            speaker = row["SPEAKER"]
            dialect = row["DIALECT"]
            trans_dialect = row["TRANSCRIPTION_DIALECT"]
            trans_german = row["TRANSCRIPTION_GERMAN"]
            audio_file = row["FILEPATH"]  # e.g., data_processed/audio/xxx_0.wav

            # Resolve absolute audio path
            # FILEPATH is relative to betthupferl root
            wav_path = BETTHUPFERL_ROOT / audio_file
            if not wav_path.exists():
                pred = f"FILE_NOT_FOUND: {wav_path}"
            else:
                pred = transcribe_with_qwen3_asr(model, str(wav_path))

            output_rows.append({
                "INDEX": idx,
                "SPEAKER": speaker,
                "DIALECT": dialect,
                "TRANSCRIPTION_DIALECT": trans_dialect,
                "TRANSCRIPTION_GERMAN": trans_german,
                "TRANSCRIPTION_MODEL": pred,
            })

        # Write per-story TSV (matching existing format)
        out_df = pd.DataFrame(output_rows)
        out_path = out_dir / f"{story}.tsv"
        out_df.to_csv(out_path, sep="\t", index=False)

    # --- Write aggregate files (matching existing format) ---
    _write_aggregates(out_dir, df)

    print(f"\n[DONE] Qwen3-ASR {model_key} inference complete.")
    print(f"Results saved to: {out_dir}")


def _write_aggregates(out_dir: Path, df: pd.DataFrame):
    """Write aggregate prediction files matching the existing Betthupferl format."""
    # Read all predictions back
    all_rows = []
    for tsv_path in sorted(out_dir.glob("*.tsv")):
        try:
            pred_df = pd.read_csv(tsv_path, sep="\t")
            all_rows.append(pred_df)
        except Exception:
            continue

    if not all_rows:
        return

    all_preds = pd.concat(all_rows, ignore_index=True)

    # Separate dialect-only and hochdeutsch
    dialect_mask = ~all_preds["DIALECT"].str.startswith("hochdeutsch")
    hd_mask = all_preds["DIALECT"].str.startswith("hochdeutsch")

    # Dialect predictions (raw)
    dialect_texts = all_preds[dialect_mask]["TRANSCRIPTION_MODEL"].tolist()
    with open(out_dir / "alldialects_pred.txt", "w") as f:
        f.write("\n".join(dialect_texts))

    # Hochdeutsch predictions
    hd_texts = all_preds[hd_mask]["TRANSCRIPTION_MODEL"].tolist()
    if hd_texts:
        with open(out_dir / "allhochdeutsch_pred.txt", "w") as f:
            f.write("\n".join(hd_texts))

    print(f"  Aggregate files written: alldialects_pred.txt ({len(dialect_texts)} lines)")


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR on Betthupferl")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["0.6B", "1.7B"],
        help="Model size to run"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't skip already-completed stories"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for inference (only used if model supports it)"
    )
    args = parser.parse_args()

    # --- Load model ---
    print(f"Loading Qwen3-ASR {args.model} from {MODEL_MAP[args.model]}...")
    print("This may take a few minutes on first run (model download).")

    try:
        # Try the official qwen_asr package first
        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(
            MODEL_MAP[args.model],
            device=args.device,
            max_inference_batch_size=args.batch_size,
        )
        print("[OK] Loaded via qwen_asr package")

    except ImportError:
        print("[WARN] qwen_asr not installed. Trying transformers pipeline...")
        try:
            from transformers import pipeline

            model = pipeline(
                "automatic-speech-recognition",
                model=MODEL_MAP[args.model],
                device=args.device,
            )
            print("[OK] Loaded via transformers pipeline")
        except Exception as e:
            print(f"[FATAL] Cannot load model: {e}")
            print("\nInstall either:")
            print("  pip install qwen-asr")
            print("  pip install transformers torch")
            sys.exit(1)

    run_inference(args.model, resume=not args.no_resume)


if __name__ == "__main__":
    main()
