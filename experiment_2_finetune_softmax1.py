"""
Experiment 2: Small fine-tune to compare standard softmax vs softmax-1.

Runs two short fine-tunes starting from pretrained GPT-2 weights:
- standard attention (softmax)
- softmax-1 attention

For each run, saves:
- checkpoint (.pt) if training is performed
- metrics.csv with pre- and post-finetune metrics
- figures (attention + hidden states) pre- and post-finetune
"""

from __future__ import annotations

import csv
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from metrics import (
    first_token_max_attention_rate,
    hidden_state_kurtosis_by_position,
    max_abs_activation_by_position,
    perplexity_from_mean_loss,
    split_first_token_vs_rest,
)
from model import GPT, GPTConfig
from visualizations.visu import AttentionVisualizer, compute_mean_attention, compute_mean_hidden, plot_attention, plot_hidden_states


# =========================
# Global settings (edit me)
# =========================

RUN_NAME = "exp2_finetune_softmax1"
OUT_ROOT = "runs"

MODEL_INIT = "gpt2"  # smaller is faster
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = "float16" if DEVICE == "cuda" else "float32"
COMPILE = False

DATASET = "shakespeare"  # expects data/<DATASET>/{train.bin,val.bin,meta.pkl?}
BLOCK_SIZE = 128
BATCH_SIZE = 4

MAX_ITERS = 200
EVAL_INTERVAL = 50
EVAL_ITERS = 50
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

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


def _get_data_memmap(dataset: str, split: str) -> np.memmap:
    data_dir = os.path.join("data", dataset)
    path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run the corresponding data/*/prepare.py first.")
    return np.memmap(path, dtype=np.uint16, mode="r")


def _get_batch(dataset: str, split: str, block_size: int, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data = _get_data_memmap(dataset, split)
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def _estimate_loss(model: GPT, dataset: str, block_size: int, batch_size: int, device: str, dtype: str, eval_iters: int) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = _get_batch(dataset, "val", block_size, batch_size, device)
        with _device_autocast_ctx(device, dtype):
            _, loss = model(x, y)
        losses.append(float(loss.detach().cpu().item()))
    model.train()
    return float(sum(losses) / len(losses))


def _collect(model: GPT, x: torch.Tensor, device: str, dtype: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    viz = AttentionVisualizer()
    viz.register(model)
    viz.clear()
    model.eval()
    with torch.no_grad():
        with _device_autocast_ctx(device, dtype):
            _ = model(x)
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


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _save_checkpoint(path: str, model: GPT, optimizer: torch.optim.Optimizer, iter_num: int, config: Dict[str, object]) -> None:
    checkpoint = {
        "model_args": {
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embd": model.config.n_embd,
            "block_size": model.config.block_size,
            "bias": model.config.bias,
            "vocab_size": model.config.vocab_size,
            "dropout": model.config.dropout,
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": None,
        "config": config,
    }
    torch.save(checkpoint, path)


def _maybe_load_meta_vocab_size(dataset: str) -> Optional[int]:
    meta_path = os.path.join("data", dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return int(meta.get("vocab_size")) if "vocab_size" in meta else None


def _run_variant(
    *,
    run_dir: str,
    variant_name: str,
    use_softmax1: bool,
    fixed_eval_batch: torch.Tensor,
) -> List[Dict[str, object]]:
    model = GPT.from_pretrained(MODEL_INIT, dict(dropout=0.0))
    if BLOCK_SIZE < model.config.block_size:
        model.crop_block_size(BLOCK_SIZE)

    meta_vocab_size = _maybe_load_meta_vocab_size(DATASET)
    if meta_vocab_size is not None and meta_vocab_size != model.config.vocab_size:
        pass

    model.config.use_softmax1 = use_softmax1
    model.to(DEVICE)

    if COMPILE and DEVICE == "cuda":
        model = torch.compile(model)

    optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, BETAS, "cuda" if DEVICE == "cuda" else "cpu")

    rows: List[Dict[str, object]] = []

    pre_val_loss = _estimate_loss(model, DATASET, BLOCK_SIZE, BATCH_SIZE, DEVICE, DTYPE, EVAL_ITERS)
    pre_ppl = perplexity_from_mean_loss(pre_val_loss)
    att_pre, hid_pre = _collect(model, fixed_eval_batch.to(DEVICE), DEVICE, DTYPE)
    pre_metrics = _compute_metrics(att_pre, hid_pre)
    rows.append(
        {
            "variant": variant_name,
            "phase": "pre_finetune",
            "use_softmax1": use_softmax1,
            "val_loss": pre_val_loss,
            "val_ppl": pre_ppl,
            **pre_metrics,
        }
    )

    if SAVE_PLOTS:
        plot_attention(
            compute_mean_attention(att_pre),
            compute_mean_attention(att_pre),
            save_path=os.path.join(run_dir, f"{variant_name}_pre_attention.png"),
        )
        plot_hidden_states(
            compute_mean_hidden(hid_pre),
            compute_mean_hidden(hid_pre),
            save_path=os.path.join(run_dir, f"{variant_name}_pre_hidden_states.png"),
        )

    model.train()
    for it in range(1, MAX_ITERS + 1):
        x, y = _get_batch(DATASET, "train", BLOCK_SIZE, BATCH_SIZE, DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with _device_autocast_ctx(DEVICE, DTYPE):
            _, loss = model(x, y)
        loss.backward()
        if GRAD_CLIP and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
            val_loss = _estimate_loss(model, DATASET, BLOCK_SIZE, BATCH_SIZE, DEVICE, DTYPE, EVAL_ITERS)
            val_ppl = perplexity_from_mean_loss(val_loss)
            rows.append(
                {
                    "variant": variant_name,
                    "phase": f"iter_{it}",
                    "use_softmax1": use_softmax1,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                }
            )

    ckpt_path = os.path.join(run_dir, f"{variant_name}_ckpt.pt")
    _save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        MAX_ITERS,
        {
            "dataset": DATASET,
            "block_size": BLOCK_SIZE,
            "batch_size": BATCH_SIZE,
            "max_iters": MAX_ITERS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": BETAS,
            "grad_clip": GRAD_CLIP,
            "model_init": MODEL_INIT,
            "use_softmax1": use_softmax1,
            "device": DEVICE,
        },
    )

    post_val_loss = _estimate_loss(model, DATASET, BLOCK_SIZE, BATCH_SIZE, DEVICE, DTYPE, EVAL_ITERS)
    post_ppl = perplexity_from_mean_loss(post_val_loss)
    att_post, hid_post = _collect(model, fixed_eval_batch.to(DEVICE), DEVICE, DTYPE)
    post_metrics = _compute_metrics(att_post, hid_post)
    rows.append(
        {
            "variant": variant_name,
            "phase": "post_finetune",
            "use_softmax1": use_softmax1,
            "val_loss": post_val_loss,
            "val_ppl": post_ppl,
            **post_metrics,
        }
    )

    if SAVE_PLOTS:
        plot_attention(
            compute_mean_attention(att_post),
            compute_mean_attention(att_post),
            save_path=os.path.join(run_dir, f"{variant_name}_post_attention.png"),
        )
        plot_hidden_states(
            compute_mean_hidden(hid_post),
            compute_mean_hidden(hid_post),
            save_path=os.path.join(run_dir, f"{variant_name}_post_hidden_states.png"),
        )

    return rows


def main() -> None:
    run_dir = _make_run_dir()

    fixed_x, _ = _get_batch(DATASET, "val", BLOCK_SIZE, batch_size=1, device="cpu")
    fixed_x = fixed_x.to(torch.long)

    all_rows: List[Dict[str, object]] = []
    all_rows.extend(_run_variant(run_dir=run_dir, variant_name="standard", use_softmax1=False, fixed_eval_batch=fixed_x))
    all_rows.extend(_run_variant(run_dir=run_dir, variant_name="softmax1", use_softmax1=True, fixed_eval_batch=fixed_x))

    _write_csv(os.path.join(run_dir, CSV_FILENAME), all_rows)
    with open(os.path.join(run_dir, "run_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"MODEL_INIT={MODEL_INIT}\nDATASET={DATASET}\nDEVICE={DEVICE}\nDTYPE={DTYPE}\n")
        f.write(f"BLOCK_SIZE={BLOCK_SIZE}\nBATCH_SIZE={BATCH_SIZE}\nMAX_ITERS={MAX_ITERS}\n")
        f.write(f"EVAL_INTERVAL={EVAL_INTERVAL}\nEVAL_ITERS={EVAL_ITERS}\n")
        f.write(f"LEARNING_RATE={LEARNING_RATE}\nWEIGHT_DECAY={WEIGHT_DECAY}\nBETAS={BETAS}\nGRAD_CLIP={GRAD_CLIP}\n")

    print(f"Wrote outputs to: {run_dir}")


if __name__ == "__main__":
    main()

