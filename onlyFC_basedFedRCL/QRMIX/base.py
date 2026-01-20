#!/usr/bin/env python
# coding: utf-8
"""
PyTorch implementation of FC-only subspace iteration (QR-style) to extract
top-k eigenpairs of the Hessian for the last fully-connected layer, matching
the closed-form JAX logic (no autograd Hessian).

Key points (mirrors ResNet_Only_FC_CIFAR100/ResNet20/qr_mix.py):
  - Restrict to the final nn.Linear layer (weight, bias).
  - Compute Hessian·v using the softmax-cross-entropy closed form with cached
    FC inputs (features) and probabilities; no second-order autograd.
  - Run power iterations with QR, then Rayleigh-Ritz to get eigenpairs.

Inputs
  model: nn.Module (expects a final nn.Linear classifier)
  state_dict: broadcast global parameters to load
  x, y: local tensors on the target device
  k: target rank
  s: oversampling
  q: number of power iterations
Outputs
  evals: top-k eigenvalues (torch.Tensor, [k])
  evecs: eigenvectors in FC parameter space (torch.Tensor, [p, k]), p = #FC params

Note: standalone helper; hook it into client-side logic before local training.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Dict


def _get_last_linear(model: nn.Module) -> nn.Linear:
    """Return the last nn.Linear module (assumed classifier)."""
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise ValueError("Model does not contain an nn.Linear layer.")
    return linear_layers[-1]


def _flatten_fc_params(weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Flatten FC params in JAX order: bias first, then weight.
    """
    return torch.cat([bias.flatten(), weight.flatten()])


def _unflatten_fc(vec: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse of _flatten_fc_params: first |bias| entries, then weight.
    """
    b_numel = bias.numel()
    b_flat = vec[:b_numel].view_as(bias)
    w_flat = vec[b_numel:].view_as(weight)
    return w_flat, b_flat


def _extract_logits_and_features(model: nn.Module, x: torch.Tensor, fc: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass with a hook to grab the FC input (features).
    Supports model outputs as logits tensor or dict with "logit"/"logits" keys.
    """
    feat_holder = {}

    def hook(module, inp, output):
        # inp[0]: features before FC, shape [N, D]
        feat_holder["feat"] = inp[0].detach()

    handle = fc.register_forward_hook(hook)
    with torch.no_grad():
        out = model(x)
    handle.remove()

    if isinstance(out, dict):
        logits = out.get("logit", None)
        if logits is None:
            logits = out.get("logits", None)
        if logits is None:
            raise ValueError("Model forward dict must have 'logit' or 'logits'.")
        features = out.get("feat", None)
        if features is None:
            features = out.get("features", None)
        if features is None:
            features = feat_holder.get("feat", None)
    else:
        logits = out
        features = feat_holder.get("feat", None)

    if features is None:
        raise ValueError("Could not capture FC input features; ensure model uses the hooked FC layer.")
    return logits.detach(), features


def hvp_fc_closed_form(model: nn.Module,
                       fc: nn.Linear,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       v_flat: torch.Tensor,
                       block_size: int = 256) -> torch.Tensor:
    """
    Hessian-vector product for the FC layer using closed-form softmax CE Hessian
    (matches qr_mix.py logic). No autograd; uses cached features/probs.
    """
    device = x.device
    logits, features = _extract_logits_and_features(model, x, fc)
    probs = torch.softmax(logits, dim=1)

    v_w, v_b = _unflatten_fc(v_flat, fc.weight, fc.bias)

    N = x.size(0)
    total_res_w = torch.zeros_like(fc.weight, device=device)
    total_res_b = torch.zeros_like(fc.bias, device=device)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        feat_batch = features[start:end]    # [B, D]
        prob_batch = probs[start:end]       # [B, C]

        z_prime = torch.matmul(feat_batch, v_w.t()) + v_b  # [B, C]
        p_dot_z = torch.sum(prob_batch * z_prime, dim=1, keepdim=True)  # [B, 1]
        u = prob_batch * z_prime - prob_batch * p_dot_z                 # [B, C]

        res_w = torch.matmul(u.t(), feat_batch)            # [C, D] matches fc.weight
        res_b = torch.sum(u, dim=0)                        # [C]

        total_res_w += res_w
        total_res_b += res_b

    total_res_w /= float(N)
    total_res_b /= float(N)

    return _flatten_fc_params(total_res_w, total_res_b)


@torch.no_grad()
def qr_subspace_iteration_fc(model: nn.Module,
                             state_dict: Dict,
                             x: torch.Tensor,
                             y: torch.Tensor,
                             k: int,
                             s: int = 10,
                             q: int = 1,
                             loss_fn=nn.CrossEntropyLoss()) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Power iteration with QR on FC parameters only.
    Returns top-k eigenvalues and eigenvectors (FC subspace).
    """
    device = x.device
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.train(False)

    fc = _get_last_linear(model)
    p = fc.weight.numel() + fc.bias.numel()

    # Initialize random subspace
    V = torch.randn(p, k + s, device=device)
    V, _ = torch.linalg.qr(V, mode='reduced')

    # Power iterations
    for _ in range(q):
        W_cols = []
        for i in range(V.shape[1]):
            v_col = V[:, i]
            hv = hvp_fc_closed_form(model, fc, x, y, v_col)
            W_cols.append(hv)
        W = torch.stack(W_cols, dim=1)
        V, _ = torch.linalg.qr(W, mode='reduced')

    # Rayleigh-Ritz on the subspace
    HV_cols = []
    for i in range(V.shape[1]):
        hv = hvp_fc_closed_form(model, fc, x, y, V[:, i])
        HV_cols.append(hv)
    HV = torch.stack(HV_cols, dim=1)
    S = (V.T @ HV + HV.T @ V) * 0.5

    evals, evecs = torch.linalg.eigh(S)
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Keep top-k
    evals_k = evals[:k]
    evecs_k = evecs[:, :k]
    # Map back to original space
    evecs_full = V @ evecs_k
    return evals_k.detach(), evecs_full.detach()
