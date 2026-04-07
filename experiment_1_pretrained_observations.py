"""
Experiment 1: Observe paper phenomena on pretrained GPT-2 weights (no training).

Outputs (in one run folder):
- figures: attention + hidden states (standard vs softmax-1)
- metrics.csv: PPL (optional), %FirstAttn, kurtosis, max|activation| splits
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import tiktoken

from metrics import (
    first_token_max_attention_rate,
    hidden_state_kurtosis_by_position,
    max_abs_activation_by_position,
    split_first_token_vs_rest,
)
from model import GPT, GPTConfig
from visualizations.visu import AttentionVisualizer, compute_mean_attention, compute_mean_hidden, plot_attention, plot_hidden_states


# =========================
# Global settings (edit me)
# =========================

RUN_NAME = "exp1_pretrained"
OUT_ROOT = "runs"

MODEL_INIT = "gpt2-medium"  # "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = "float16" if DEVICE == "cuda" else "float32"
COMPILE = False

BLOCK_SIZE = 128
BATCH_SIZE = 1

PROMPT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote valley.",
    "Write a short dialogue between a teacher and a student about attention mechanisms.",
]

SAVE_PLOTS = True
CSV_FILENAME = "metrics.csv"


def _make_run_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_ROOT, f"{RUN_NAME}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _device_autocast_ctx(device: str, dtype: str):
    if device == "cuda":
        ptdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        return torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    return torch.autocast("cpu", enabled=False)


def _build_tokens() -> torch.Tensor:
    enc = tiktoken.get_encoding("gpt2")
    ids: List[int] = []
    for t in PROMPT_TEXTS:
        ids.extend(enc.encode(t))
        ids.append(enc.eot_token)
    if len(ids) < BLOCK_SIZE:
        ids = (ids * ((BLOCK_SIZE // max(len(ids), 1)) + 1))[:BLOCK_SIZE]
    else:
        ids = ids[:BLOCK_SIZE]
    x = torch.tensor(ids, dtype=torch.long).view(1, -1).repeat(BATCH_SIZE, 1)
    return x


def _collect(model: GPT, x: torch.Tensor, device: str, dtype: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    viz = AttentionVisualizer()
    viz.register(model)
    viz.clear()
    model.eval()
    model.to(device)
    with torch.no_grad():
        with _device_autocast_ctx(device, dtype):
            _ = model(x.to(device))
    attn = viz.attention_weights
    hids = viz.hidden_states
    viz.remove()
    return attn, hids


def _compute_metrics(attn_layers: List[torch.Tensor], hid_layers: List[torch.Tensor]) -> Dict[str, float]:
    rate = first_token_max_attention_rate(attn_layers, first_token_index=0, causal=True)

    k_by_pos = hidden_state_kurtosis_by_position(hid_layers)
    k_first, k_rest = split_first_token_vs_rest(k_by_pos, first_token_index=0)

    m_by_pos = max_abs_activation_by_position(hid_layers)
    m_first, m_rest = split_first_token_vs_rest(m_by_pos, first_token_index=0)

    return {
        "first_token_max_attention_rate": rate,
        "kurtosis_first_token": k_first,
        "kurtosis_other_tokens_mean": k_rest,
        "max_abs_activation_first_token": m_first,
        "max_abs_activation_other_tokens_mean": m_rest,
    }


def _write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    run_dir = _make_run_dir()
    x = _build_tokens()

    model_std = GPT.from_pretrained(MODEL_INIT, dict(dropout=0.0))
    if BLOCK_SIZE < model_std.config.block_size:
        model_std.crop_block_size(BLOCK_SIZE)
    model_std.config.use_softmax1 = False

    cfg = GPTConfig(
        block_size=model_std.config.block_size,
        vocab_size=model_std.config.vocab_size,
        n_layer=model_std.config.n_layer,
        n_head=model_std.config.n_head,
        n_embd=model_std.config.n_embd,
        dropout=0.0,
        bias=model_std.config.bias,
        use_softmax1=True,
        use_orthoadam=False,
    )
    model_s1 = GPT(cfg)
    model_s1.load_state_dict(model_std.state_dict(), strict=True)

    if COMPILE and DEVICE == "cuda":
        model_std = torch.compile(model_std)
        model_s1 = torch.compile(model_s1)

    att_std, hid_std = _collect(model_std, x, DEVICE, DTYPE)
    att_s1, hid_s1 = _collect(model_s1, x, DEVICE, DTYPE)

    metrics_rows: List[Dict[str, float]] = []
    metrics_rows.append({"model": MODEL_INIT, "variant": "standard", "block_size": float(BLOCK_SIZE), **_compute_metrics(att_std, hid_std)})
    metrics_rows.append({"model": MODEL_INIT, "variant": "softmax-1", "block_size": float(BLOCK_SIZE), **_compute_metrics(att_s1, hid_s1)})

    csv_path = os.path.join(run_dir, CSV_FILENAME)
    _write_csv(csv_path, metrics_rows)

    if SAVE_PLOTS:
        mean_att_std = compute_mean_attention(att_std)
        mean_att_s1 = compute_mean_attention(att_s1)
        plot_attention(mean_att_std, mean_att_s1, save_path=os.path.join(run_dir, "figure_attention.png"))

        mean_hid_std = compute_mean_hidden(hid_std)
        mean_hid_s1 = compute_mean_hidden(hid_s1)
        plot_hidden_states(mean_hid_std, mean_hid_s1, save_path=os.path.join(run_dir, "figure_hidden_states.png"))

    with open(os.path.join(run_dir, "run_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"MODEL_INIT={MODEL_INIT}\nDEVICE={DEVICE}\nDTYPE={DTYPE}\nBLOCK_SIZE={BLOCK_SIZE}\nBATCH_SIZE={BATCH_SIZE}\n")
        f.write(f"PROMPT_TEXTS={PROMPT_TEXTS}\n")
        f.write(f"model_config={asdict(model_std.config)}\n")

    print(f"Wrote outputs to: {run_dir}")


if __name__ == "__main__":
    main()

