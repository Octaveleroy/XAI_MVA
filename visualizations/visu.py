"""
Reproduce the visualization of the paper
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from contextlib import nullcontext

class AttentionVisualizer:
    """
    Capture hooks of the model to capture attention weights and hidden states after each block.
    """
    def __init__(self):
        """
        Init with list of the attention and hidden states
        """
        self.attention_weights = []  
        self.hidden_states     = []   
        self._hooks            = []

    def register(self, model):
        """Save hooks of the model"""
        for block in model.transformer.h:

            def make_attn_hook(module, input, output):
                # output = y après attn_dropout ; on veut att
                # On re-calcule att depuis les poids déjà capturés
                pass

            def make_block_hook(block_module):
                def hook_fn(module, input, output):
                    self.hidden_states.append(output[0].detach().cpu()
                                              if isinstance(output, tuple)
                                              else output.detach().cpu())
                return hook_fn

            # Register the forward hook and add hidden states
            h = block.register_forward_hook(make_block_hook(block))
            self._hooks.append(h)

        
        # Capture the attention for each block
        for block in model.transformer.h:
            def make_attn_capture(attn_module):
                original_forward = attn_module.forward

                def patched_forward(x):
                    """
                    Forwad of the block but patched
                    """
                    B, T, C = x.size()
                    q, k, v = attn_module.c_attn(x).split(attn_module.n_embd, dim=2)
                    nh = attn_module.n_head
                    hs = C // nh
                    k = k.view(B, T, nh, hs).transpose(1, 2)
                    q = q.view(B, T, nh, hs).transpose(1, 2)
                    v = v.view(B, T, nh, hs).transpose(1, 2)

                    import math, torch.nn.functional as F

                    # Q.K^T/sqrt(hs)
                    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))

                    # Causal Mask
                    if hasattr(attn_module, 'bias'):
                        att = att.masked_fill(
                            attn_module.bias[:, :, :T, :T] == 0, float('-inf')
                        )

                    # Softmax standard ou softmax-1 of the attention
                    if getattr(attn_module, 'use_softmax1', False):
                        att_max = att.detach().max(dim=-1, keepdim=True).values
                        att_max = torch.clamp(att_max, min=0)
                        att_shifted = att - att_max
                        numerator = torch.exp(att_shifted)
                        denominator = torch.exp(-att_max) + numerator.sum(dim=-1, keepdim=True)
                        att_prob = numerator / denominator
                    else:
                        att_prob = F.softmax(att, dim=-1)
                    
                    # Add attention weights for each tokens of the block
                    self.attention_weights.append(att_prob.detach().cpu())
                    
                    # Values
                    att_prob = attn_module.attn_dropout(att_prob)
                    y = att_prob @ v
                    y = y.transpose(1, 2).contiguous().view(B, T, C)
                    y = attn_module.resid_dropout(attn_module.c_proj(y))
                    return y

                attn_module.forward = patched_forward

            make_attn_capture(block.attn)

    def clear(self):
        """
        Clear the list of attention and hidden states
        """
        self.attention_weights = []
        self.hidden_states     = []

    def remove(self):
        """
        Remove the hooks of the models
        """
        for h in self._hooks:
            h.remove()
        self._hooks = []


def compute_mean_attention(attention_weights):
    """
    Compute mean attention on all layers and all heads
    """
    stacked = torch.stack(attention_weights, dim=0)   # (L, B, nh, T, T)
    mean_att = stacked.mean(dim=(0, 1, 2))            # (T, T)
    return mean_att.numpy()


def compute_mean_hidden(hidden_states):
    """
    Compute mean of the hidden stats on all the layer and absolute values. 
    """
    stacked = torch.stack(hidden_states, dim=0)       # (L, B, T, C)
    mean_h  = stacked.mean(dim=(0, 1)).abs()          # (T, C)
    return mean_h.numpy()




def plot_attentions_keys(attn_standard, attn_softmax1,save_path="results/attentions_keys.png"): 
    """
    Plot the activation of the layers
    """

    fig = plt.figure(figsize=(8,8))
    fig.patch.set_facecolor('white') 



def plot_attention(
    attn_standard, attn_softmax1,
    save_path="result/figure_attention.png"
):
    """
    Mean attention
    """
    T = attn_standard.shape[0]
    attn_data  = [attn_standard, attn_softmax1]
    row_titles = ["Current Transformer Models", "Our Transformer Models"]

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    fig.patch.set_facecolor('white')

    for row, ax in enumerate(axes):
        im = ax.imshow(
            attn_data[row],
            cmap='Blues_r',
            aspect='auto',
            vmin=0, vmax=1,
            origin='upper'
        )
        ax.set_title(row_titles[row].upper(), fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel("Key Position",   fontsize=9)
        ax.set_ylabel("Query Position", fontsize=9)
        ax.tick_params(labelsize=8)

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)


        if row == 0:
            rect = plt.Rectangle(
                (-0.5, -0.5), 1, T,
                linewidth=1.5, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Save attention figure to {save_path}")
    plt.show()


def plot_hidden_states(
    hidden_standard, hidden_softmax1,
    save_path="result/figure_hidden_states.png"
):
    """
    Mean activations of hidden states
    """
    T = hidden_standard.shape[0]
    hidden_data = [hidden_standard, hidden_softmax1]
    row_titles  = ["Current Transformer Models", "Our Transformer Models"]

    vmin_h   = max(hidden_standard.min(), 1e-2)
    vmax_h   = max(hidden_standard.max(), hidden_softmax1.max())
    log_norm = LogNorm(vmin=vmin_h, vmax=vmax_h)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor('white')

    for row, ax in enumerate(axes):
        im = ax.imshow(
            hidden_data[row],
            cmap='Blues_r',
            aspect='auto',
            norm=log_norm,
            origin='upper'
        )
        ax.set_title(row_titles[row].upper(), fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel("Channel ID",     fontsize=9)
        ax.set_ylabel("Token Position", fontsize=9)
        ax.tick_params(labelsize=8)

        cb = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
        cb.ax.tick_params(labelsize=8)

        if row == 0:
            outlier_ch = int(np.argmax(hidden_data[row].max(axis=0)))
            rect = plt.Rectangle(
                (outlier_ch - 1.5, -0.5), 3, T,
                linewidth=1.5, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Save hidden states to {save_path}")
    plt.show()

def run_model_and_collect(model, tokens, device='cpu'):
    """
    Forward pass of the model and collect attention and hidden states. 

    :param model: GPT nanoGPT
    :type model: nanoGPT 
    :param tokens: tokens ids of a sequence
    :type tokens: torch.Tensor
    :param device: cpu or cuda
    :type device: str


    :returns: 
    :rtype: 
    Returns
    -------
    mean_attn   : np.array (T, T)
    mean_hidden : np.array (T, C)
    """
    viz = AttentionVisualizer()
    viz.register(model)
    viz.clear()

    model.eval()
    with torch.no_grad():
        with (torch.amp.autocast(device_type=device)
              if device == 'cuda' else nullcontext()):
            _ = model(tokens.to(device))

    mean_attn   = compute_mean_attention(viz.attention_weights)
    mean_hidden = compute_mean_hidden(viz.hidden_states)

    viz.remove()
    return mean_attn, mean_hidden


if __name__ == "__main__":
    import sys
    import os

    NANOGPT_PATH = "." 
    sys.path.insert(0, NANOGPT_PATH)

    from model import GPT, GPTConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    cfg_standard = GPTConfig(
        n_layer=12, n_head=12, n_embd=768,
        block_size=128, vocab_size=50257, dropout=0.0,
        bias=True,
        use_softmax1=False 
    )
    cfg_softmax1 = GPTConfig(
        n_layer=12, n_head=12, n_embd=768,
        block_size=128, vocab_size=50257, dropout=0.0,
        bias=True,
        use_softmax1=True   
    )

    # Random init just to test visu
    model_standard = GPT(cfg_standard).to(device)
    model_softmax1 = GPT(cfg_softmax1).to(device)

    # Option B : charger des checkpoints entraînés
    # ckpt = torch.load('out/ckpt.pt', map_location=device)
    # model_standard.load_state_dict(ckpt['model'])

    T = 128 
    torch.manual_seed(42)
    tokens = torch.randint(0, 50257, (1, T))

    print("Forward pass standard model...")
    attn_std, hidden_std = run_model_and_collect(model_standard, tokens, device)

    print("Forward pass softmax-1...")
    attn_sm1, hidden_sm1 = run_model_and_collect(model_softmax1, tokens, device)

    plot_attention(
    attn_standard=attn_std,
    attn_softmax1=attn_sm1,
    save_path="result/figure_attention.png"
    )

    plot_hidden_states(
        hidden_standard=hidden_std,
        hidden_softmax1=hidden_sm1,
        save_path="result/figure_hidden_states.png"
    )


