#!/usr/bin/env python3
"""
Experiment 3: Temperature sweep — continuous modulation of decoder LM prior strength.

  temperature = 0.0  → greedy decoding, strongest LM prior
  temperature = 0.2  → mild mixing
  temperature = 0.5  → balanced
  temperature = 0.8  → more diverse, weaker prior
  temperature = 1.0  → most diverse, weakest prior

Optimized for speed:
  - Subsampled to 5,000 sentences
  - Disabled VAD filter (vad_filter=False)
  - Pure Greedy decoding (beam_size=1, best_of=1)
  - Buffer I/O writes every 50 samples
"""

import argparse, os, sys, time
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA   = PROJECT_ROOT / "master_metadata.csv"
AUDIO_DIR  = PROJECT_ROOT / "data1.1"
OUT_DIR    = PROJECT_ROOT / "results" / "swissdial"

TEMPERATURES = [0.0, 0.2, 0.5, 0.8, 1.0]

# Standardized compute types per model
COMPUTE_TYPE = {
    "tiny":     "float16",
    "small":    "float16",
    "medium":   "float16",
    "large-v3": "int8_float16",
}


def resolve_audio(rel_path: str) -> str:
    filename = os.path.basename(rel_path)
    dialect = rel_path.split("/")[1]
    return str(AUDIO_DIR / dialect / filename)


def load_and_prep(test_mode: bool, max_samples: int = 5000) -> pd.DataFrame:
    df = pd.read_csv(METADATA)
    df["resolved_audio"] = df["audio_path"].apply(resolve_audio)
    
    if not os.path.exists(df["resolved_audio"].iloc[0]):
        sys.exit(f"Audio not found: {df['resolved_audio'].iloc[0]}")
        
    if test_mode:
        df = df.head(200)
        print(f"[TEST MODE] 200 sentences selected")
    elif max_samples and len(df) > max_samples:
        # 直接截取前 5,000 条，保持与 Exp 2 截取策略一致
        df = df.head(max_samples)
        print(f"[SUBSET] Cut dataset to first {len(df)} sentences")
        
    return df


def get_output_path(model_key: str) -> Path:
    return OUT_DIR / f"master_metadata_whisper_{model_key.replace('-','_')}_temp_sweep.csv"


def run_temp_sweep(model_key: str, test_mode: bool = False):
    from faster_whisper import WhisperModel

    size    = model_key
    ct      = COMPUTE_TYPE[size]
    out_csv = get_output_path(model_key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prep(test_mode, max_samples=5000)

    # Resume logic
    done = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done = set(existing["sentence_id"].astype(str))
        if done:
            print(f"[RESUME] {len(done)} already done in CSV")

    df_todo = df[~df["sentence_id"].astype(str).isin(done)]
    if len(df_todo) == 0:
        return print("[DONE] All complete.")

    print(f"Loading faster-whisper '{size}' (compute_type={ct})...", flush=True)
    # 给 CPU 线程和 num_workers 提速
    model = WhisperModel(size, device="cuda", compute_type=ct,
                         num_workers=4, cpu_threads=4)

    base = size.replace("-", "_")
    temp_cols = [f"whisper_{base}_T{str(t).replace('.','_')}" for t in TEMPERATURES]
    cols = ["sentence_id", "dialect", "audio_path",
            "ref_high_german", "ref_swiss_german"] + temp_cols

    write_header = not out_csv.exists()
    if write_header:
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False)

    batch_buffer = []
    t0 = time.time()
    
    for i, (_, row) in enumerate(df_todo.iterrows()):
        af = row["resolved_audio"]
        rec = {
            "sentence_id":       row["sentence_id"],
            "dialect":           row["dialect"],
            "audio_path":        row["audio_path"],
            "ref_high_german":   row["ref_high_german"],
            "ref_swiss_german": row["ref_swiss_german"],
        }
        
        # 依次运行 5 个温度
        for j, T in enumerate(TEMPERATURES):
            segs, _ = model.transcribe(
                af,
                beam_size=1,        # 【关键提速】切回 Greedy，速度提升 3-5 倍
                best_of=1,          # 【关键提速】禁用多路候选采样
                temperature=T,
                language="de",
                vad_filter=False,   # 【关键提速】禁用 VAD 过滤器
            )
            rec[temp_cols[j]] = " ".join(s.text.strip() for s in segs)

        batch_buffer.append(rec)

        # 每 50 条数据批量写入一次 CSV，减少磁盘 I/O 拖累
        if len(batch_buffer) >= 50 or (i + 1) == len(df_todo):
            pd.DataFrame(batch_buffer).to_csv(out_csv, mode="a", index=False, header=False)
            batch_buffer.clear()

        # 进度与 ETA 输出
        if (i + 1) % 50 == 0 or (i + 1) == len(df_todo):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df_todo) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{len(df_todo)}] {rate:.2f} utt/s (×5 temps), ETA {eta:.1f} min", flush=True)

    print(f"[DONE] Saved to {out_csv.name}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Whisper temperature sweep")
    p.add_argument("--model", required=True, choices=["tiny","small","medium","large-v3"])
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    print("=" * 60)
    print("Experiment 3: Temperature sweep (Optimized)")
    print(f"  Model: {args.model}")
    print(f"  Temperatures: {TEMPERATURES}")
    print("=" * 60)
    run_temp_sweep(args.model, test_mode=args.test)


if __name__ == "__main__":
    main()