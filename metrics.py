"""
Metrics used in: "From Attention to Activation: Unravelling the Enigmas of Large Language Models"
(Kaul et al., 2024, arXiv:2410.17174v1).

This module provides one function per metric, designed to work with tensors you can
already collect in this repo (e.g., `visualizations/visu.py`).
"""

from __future__ import annotations


from typing import Sequence, Tuple, Union

import math

import torch


TensorLike = Union[torch.Tensor]


def perplexity_from_mean_loss(mean_loss: Union[float, torch.Tensor]) -> float:
    """
    Compute perplexity (PPL) from an average cross-entropy loss.

    Why it matters:
        PPL is the standard language modeling quality metric. The paper uses PPL to
        verify that mitigating the two phenomena does not harm model performance.

    Paper reference:
        Section 5 "EXPERIMENTS", "Metrics" paragraph (PDF page --7 of 46--),
        where PPL is listed as metric (1).

    Args:
        mean_loss: Average negative log-likelihood / cross-entropy (natural log).

    Returns:
        Perplexity as a Python float.
    """
    if isinstance(mean_loss, torch.Tensor):
        mean_loss_val = float(mean_loss.detach().cpu().item())
    else:
        mean_loss_val = float(mean_loss)
    return float(math.exp(mean_loss_val))


def first_token_max_attention_rate(
    attention_probs_by_layer: Sequence[torch.Tensor],
    *,
    first_token_index: int = 0,
    causal: bool = True,
) -> float:
    """
    Percentage of (query, head) pairs whose most-attended key is the first token.

    Definition (as used in the paper):
        For each layer and head, and for each query position, find the key position
        with maximum attention probability. Count how often that argmax equals the
        first token position.

    Why it matters:
        This quantifies the "first token dominance" phenomenon in attention maps.

    Paper reference:
        Section 5 "EXPERIMENTS", "Metrics" paragraph (PDF page --7 of 46--),
        metric (4): "the percentage of (query, head) pairs ...".
        Also discussed earlier in Section 2.1 (PDF page --3 of 46--).

    Args:
        attention_probs_by_layer:
            Sequence of attention probability tensors, one per layer.
            Expected shape per tensor: (B, n_head, T, T), i.e. probabilities over keys.
            This matches what `visualizations/visu.py` stores in `attention_weights`.
        first_token_index: Which key index counts as "first token" (default 0).
        causal:
            If True, ignore future keys (j > t) for each query t when computing argmax.
            In a causal model this is the intended behavior.

    Returns:
        Rate in [0, 1] as a float.
    """
    if len(attention_probs_by_layer) == 0:
        raise ValueError("attention_probs_by_layer must be non-empty")

    total = 0
    count = 0

    for att in attention_probs_by_layer:
        if att.dim() != 4:
            raise ValueError(f"Expected attention tensor of shape (B,H,T,T), got {tuple(att.shape)}")
        B, H, Tq, Tk = att.shape
        if Tq != Tk:
            raise ValueError(f"Expected square attention over sequence, got Tq={Tq}, Tk={Tk}")
        if not (0 <= first_token_index < Tk):
            raise ValueError(f"first_token_index={first_token_index} out of range for Tk={Tk}")

        att_use = att
        if causal:
            mask = torch.triu(torch.ones((Tq, Tk), device=att.device, dtype=torch.bool), diagonal=1)
            att_use = att.masked_fill(mask.view(1, 1, Tq, Tk), float("-inf"))

        argmax_key = att_use.argmax(dim=-1)  # (B, H, T)
        total += int((argmax_key == first_token_index).sum().item())
        count += B * H * Tq

    return float(total / count)


def hidden_state_kurtosis_by_position(
    hidden_states_by_layer: Sequence[torch.Tensor],
    *,
    eps: float = 1e-12,
    unbiased: bool = False,
) -> torch.Tensor:
    """
    Kurtosis of hidden-state activations across feature dimensions, per token position.

    Definition (paper-style):
        Given hidden states X with shape (layer, batch, token, feature),
        compute kurtosis across the feature dimension for each (layer, token),
        then (optionally) average across layers.

        This function returns the per-position kurtosis averaged across layers and batch.

    Why it matters:
        High kurtosis indicates heavy tails / outlier activations in hidden states, which
        the paper links to poor quantization robustness.

    Paper reference:
        Section 4 "METHOD: OUTLIER ACTIVATIONS" introduces kurtosis (PDF page --5 of 46--),
        Eq. (2), and Section 5 "EXPERIMENTS" uses kurtosis as metric (2) (PDF page --7 of 46--).

    Args:
        hidden_states_by_layer:
            Sequence of hidden-state tensors, one per layer.
            Expected shape per tensor: (B, T, C).
            This matches what `visualizations/visu.py` stores in `hidden_states`.
        eps: Numerical stabilizer to avoid division by zero.
        unbiased:
            If True, apply a small-sample correction (not used in the paper).
            Default False matches the common "population" kurtosis formula.

    Returns:
        A tensor of shape (T,) containing kurtosis per token position.
    """
    if len(hidden_states_by_layer) == 0:
        raise ValueError("hidden_states_by_layer must be non-empty")

    hs = []
    for x in hidden_states_by_layer:
        if x.dim() != 3:
            raise ValueError(f"Expected hidden state of shape (B,T,C), got {tuple(x.shape)}")
        hs.append(x)
    X = torch.stack(hs, dim=0)  # (L, B, T, C)

    mu = X.mean(dim=-1, keepdim=True)  # (L, B, T, 1)
    centered = X - mu
    m2 = (centered ** 2).mean(dim=-1)  # (L, B, T)
    m4 = (centered ** 4).mean(dim=-1)  # (L, B, T)
    kurt = m4 / (m2 ** 2 + eps)  # (L, B, T)

    if unbiased:
        C = X.shape[-1]
        if C > 3:
            n = float(C)
            g2 = kurt - 3.0
            factor = (n - 1.0) / ((n - 2.0) * (n - 3.0))
            correction = ((n + 1.0) * g2 + 6.0) * factor
            kurt = correction + 3.0

    return kurt.mean(dim=(0, 1))  # (T,)


def max_abs_activation_by_position(
    hidden_states_by_layer: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Maximum absolute hidden-state activation across feature dimensions, per token position.

    Why it matters:
        This is a direct proxy for activation outliers. The paper reports max absolute
        activation (separately for the first token vs other tokens) alongside kurtosis.

    Paper reference:
        Section 5 "EXPERIMENTS", "Metrics" paragraph (PDF page --7 of 46--),
        where max absolute activation is included as metric (3).

    Args:
        hidden_states_by_layer:
            Sequence of hidden-state tensors, one per layer.
            Expected shape per tensor: (B, T, C).

    Returns:
        A tensor of shape (T,) containing max |activation| per token position,
        averaged over layers and batch.
    """
    if len(hidden_states_by_layer) == 0:
        raise ValueError("hidden_states_by_layer must be non-empty")

    X = torch.stack(hidden_states_by_layer, dim=0)  # (L, B, T, C)
    max_abs = X.abs().amax(dim=-1)  # (L, B, T)
    return max_abs.mean(dim=(0, 1))  # (T,)


def split_first_token_vs_rest(values_by_position: torch.Tensor, *, first_token_index: int = 0) -> Tuple[float, float]:
    """
    Convenience helper: aggregate a per-position metric into (first token, other tokens).

    Why it matters:
        The paper frequently reports metrics separately for the first token position and
        the remaining positions (e.g., kurtosis and max |activation| in Table 2).

    Paper reference:
        Table 2 formatting (PDF page --7 of 46--) reports Em[κ_{m,1}] vs Em[κ_{m,>1}]
        and analogous splits for activation values.

    Args:
        values_by_position: Tensor of shape (T,).
        first_token_index: Which index is treated as the "first token".

    Returns:
        (first_token_value, mean_other_tokens_value) as floats.
    """
    if values_by_position.dim() != 1:
        raise ValueError(f"Expected 1D tensor (T,), got {tuple(values_by_position.shape)}")
    T = values_by_position.shape[0]
    if not (0 <= first_token_index < T):
        raise ValueError(f"first_token_index={first_token_index} out of range for T={T}")
    first = float(values_by_position[first_token_index].detach().cpu().item())
    if T == 1:
        rest = float("nan")
    else:
        mask = torch.ones((T,), device=values_by_position.device, dtype=torch.bool)
        mask[first_token_index] = False
        rest = float(values_by_position[mask].mean().detach().cpu().item())
    return first, rest


def quantization_ppl_delta(ppl_full: Union[float, torch.Tensor], ppl_quantized: Union[float, torch.Tensor]) -> float:
    """
    Quantization penalty: ΔPPL = PPL(quantized) - PPL(full precision).

    Why it matters:
        The paper's practical claim is that removing attention sinks and activation outliers
        allows basic 8-bit/4-bit quantization to preserve performance. They report PPL before
        and after quantization, and the difference (penalty).

    Paper reference:
        Section 5.2 "QUANTISATION" (PDF page --8 of 46--) and Table 3 (PDF page --8 of 46--).

    Args:
        ppl_full: Perplexity of the full precision model.
        ppl_quantized: Perplexity after quantization.

    Returns:
        ΔPPL as a float.
    """
    if isinstance(ppl_full, torch.Tensor):
        ppl_full = float(ppl_full.detach().cpu().item())
    if isinstance(ppl_quantized, torch.Tensor):
        ppl_quantized = float(ppl_quantized.detach().cpu().item())
    return float(ppl_quantized - ppl_full)

