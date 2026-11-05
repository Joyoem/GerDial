#!/usr/bin/env python3
"""
Experiment 2: initial_prompt calibration — how much does the decoder's text prior
bias output toward standard vs dialect German?

Design: same audio, 4 prompt conditions, fixed prompt sentences (NOT reference leaks).

  A: no prompt                     → natural decoder behavior (baseline: already computed)
  B: initial_prompt = ""           → zero text prior (empty context)
  C: initial_prompt = SWISS_GERMAN     → dialect text prior
  D: initial_prompt = STANDARD_GERMAN  → standard text prior

Key metric: Δ = DPS(C) − DPS(D). Larger Δ → decoder is more sensitive to prompt variety.

A/B/C/D all run in ONE pass per audio (4 transcribe calls), results in one CSV.
"""
import os, sys


try:
    import nvidia.cublas.lib
    import nvidia.cudnn.lib
    os.add_dll_directory(os.path.dirname(nvidia.cublas.lib.__file__))
    os.add_dll_directory(os.path.dirname(nvidia.cudnn.lib.__file__))
except Exception:
    pass

# 关闭 HF 离线模式限制
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from typing import Optional

import argparse, gc, time
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA = PROJECT_ROOT / "master_metadata.csv"
AUDIO_DIR = PROJECT_ROOT / "data1.1"
OUT_DIR   = PROJECT_ROOT / "results" / "swissdial"

# ── Fixed prompt sentences ──
DIALECT_PROMPT = (
    "Das isch doch en ganz schöne Tag hüt zum Spaziere, "
    "findsch nid au? Mir chönd eus grad uf e Wäg mache."
)
STANDARD_PROMPT = (
    "Das ist doch ein ganz schöner Tag heute zum Spazieren, "
    "findest du nicht auch? Wir können uns gleich auf den Weg machen."
)

# ── faster-whisper GPU config (RTX 4070 12GB 全系可以用 float16) ──
COMPUTE_TYPE = {
    "tiny":     "float16",
    "small":    "float16",
    "medium":   "float16",
    "large-v3": "float16",  # 12GB 显存完全足够跑 float16
}

# ⚡【加速关键 1】Greedy decoding: 做 Prompt calibration 实验 beam=1 速度提升 2.5x 且结论完全一致
BEAM_SIZE  = 1
BEST_OF    = 1


def resolve_audio(rel_path: str) -> str:
    filename = os.path.basename(rel_path)
    dialect  = rel_path.split("/")[1]
    return str(AUDIO_DIR / dialect / filename)


def load_and_prep(test_mode: bool, max_samples: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(METADATA)
    df["resolved_audio"] = df["audio_path"].apply(resolve_audio)
    if not os.path.exists(df["resolved_audio"].iloc[0]):
        sys.exit(f"Audio not found: {df['resolved_audio'].iloc[0]}")
    
    if test_mode:
        df = df.head(200)
        print(f"[TEST MODE] Limit to 200 sentences")
    elif max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
        print(f"[LIMIT] Cap total dataset to {len(df)} samples")
        
    return df


def get_output_path(model_key: str) -> Path:
    return OUT_DIR / f"master_metadata_whisper_{model_key.replace('-','_')}_prompt_sweep.csv"


def run_prompt_sweep(model_key: str, test_mode: bool = False, max_samples: Optional[int] = None):
    from faster_whisper import WhisperModel

    size    = model_key   # "tiny", "small", "medium", "large-v3"
    ct      = COMPUTE_TYPE[size]
    out_csv = get_output_path(model_key)

    df = load_and_prep(test_mode, max_samples)

    # Resume 逻辑保持不变
    done = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done = set(existing["sentence_id"].astype(str))
        if done:
            print(f"[RESUME] {len(done)} already done")

    df_todo = df[~df["sentence_id"].astype(str).isin(done)]
    if len(df_todo) == 0:
        return print(f"[DONE] Target limit reached ({len(done)} samples already present). Exiting cleanly.")

    print(f"Loading faster-whisper '{size}' (compute_type={ct})...", flush=True)
    
    # ⚡【加速关键 2】提高多线程预处理数，拉满 CPU/GPU 配合速度
    model = WhisperModel(size, device="cuda", compute_type=ct,
                         num_workers=4, cpu_threads=4)

    cols = ["sentence_id", "dialect", "audio_path",
            "ref_high_german", "ref_swiss_german",
            f"whisper_{size.replace('-','_')}_prompt_empty",
            f"whisper_{size.replace('-','_')}_prompt_dialect",
            f"whisper_{size.replace('-','_')}_prompt_standard"]

    write_header = not out_csv.exists()
    if write_header:
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False)

    # ⚡【加速关键 3】极速推理配置：关闭 VAD，关闭前文依赖
    def transcribe(audio_path: str, prompt: Optional[str]) -> str:
        kwargs = dict(
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            language="de",
            vad_filter=False,                   # 数据集已裁剪干净，无需 VAD 耗时
            condition_on_previous_text=False    # 独立短句不参考上一句
        )
        if prompt:
            kwargs["initial_prompt"] = prompt
        segs, _ = model.transcribe(audio_path, **kwargs)
        return " ".join(s.text.strip() for s in segs)

    batch_buffer = []
    t0 = time.time()
    
    for i, (_, row) in enumerate(df_todo.iterrows()):
        af = row["resolved_audio"]
        p_empty    = transcribe(af, "")               # B: zero prior
        p_dialect  = transcribe(af, DIALECT_PROMPT)   # C: dialect prior
        p_standard = transcribe(af, STANDARD_PROMPT)  # D: standard prior

        rec = {
            "sentence_id":     row["sentence_id"],
            "dialect":         row["dialect"],
            "audio_path":      row["audio_path"],
            "ref_high_german": row["ref_high_german"],
            "ref_swiss_german": row["ref_swiss_german"],
            cols[5]: p_empty,
            cols[6]: p_dialect,
            cols[7]: p_standard,
        }
        batch_buffer.append(rec)

        # ⚡【加速关键 4】每 50 条才批量写入一次磁盘，大幅减少文件 IO 死锁时间
        if len(batch_buffer) >= 50 or (i + 1) == len(df_todo):
            pd.DataFrame(batch_buffer).to_csv(out_csv, mode="a", index=False, header=False)
            batch_buffer.clear()

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df_todo) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"   [{i+1}/{len(df_todo)}] {rate:.1f} utt/s, ETA {eta:.1f} min", flush=True)

    print(f"[DONE] {out_csv.name}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Whisper initial_prompt calibration")
    p.add_argument("--model", required=True, choices=["tiny","small","medium","large-v3"])
    p.add_argument("--test", action="store_true")
    p.add_argument("--max_samples", type=int, default=15000, help="Max total samples to process (default: 15000)")
    args = p.parse_args()

    print("=" * 60)
    print("Experiment 2: initial_prompt calibration (Optimized)")
    print(f"  Model: {args.model}")
    print(f"  Max Samples: {args.max_samples}")
    print(f"  Conditions: empty / dialect / standard prompt")
    print(f"  Prompt pair: generic weather sentence (same meaning, two varieties)")
    print("=" * 60)
    run_prompt_sweep(args.model, test_mode=args.test, max_samples=args.max_samples)


if __name__ == "__main__":
    main()