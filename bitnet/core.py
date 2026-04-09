# bitnet/core.py
# PyTorch BitNet implementation (BitLinear, absmax quantization, decoder block, model)
# Requires: torch>=1.12

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Helpers: absmax quantization (returns integer levels and scale gamma)
# -------------------------
def absmax_quantize_to_int(x, bits=8, clip_eps=1e-6, per_token=False):
    """
    Quantize tensor x to signed integer range [-Qb, Qb].
    Returns: q_int (float tensor but integer values), gamma (scale)
    If per_token=True, expects x shaped (B, T, D) and quantizes per token (over D dim).
    """
    Qb = 2 ** (bits - 1)
    if per_token and x.dim() == 3:
        # compute gamma per token (B,T,1)
        gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=clip_eps)
        scaled = x * (Qb / gamma)
        clipped = scaled.clamp(-Qb + clip_eps, Qb - clip_eps)
        q = torch.round(clipped)
        return q, gamma  # q in [-Qb, Qb], gamma shape (B,T,1)
    else:
        # per-tensor
        gamma = x.abs().amax().clamp(min=clip_eps)
        scaled = x * (Qb / gamma)
        clipped = scaled.clamp(-Qb + clip_eps, Qb - clip_eps)
        q = torch.round(clipped)
        return q, gamma  # q is tensor same shape as x, but integer-valued floats

def absmax_dequantize_int(q_int, gamma, bits=8):
    """
    Convert integer levels back to floating-point scaled values: (q/Qb) * gamma
    """
    Qb = 2 ** (bits - 1)
    return (q_int / Qb) * gamma

# -------------------------
# BitLinear (group quantization supported)
# -------------------------
class BitLinear(nn.Module):
    """
    Replaces nn.Linear in BitNet.
    weight_fp: latent full precision weight [out_features, in_features]
    Forward:
      1) compute alpha_g (mean) per group and center weights -> Wc = W - alpha_g
      2) compute beta_g = (G / (n*m)) * ||W^(g)||_1   (matches paper's group formula)
      3) Wb = Sign(Wc)  (binarized)
      4) x quantized via absmax to integer levels q_int (per-tensor during training; per-token option for inference)
      5) y_int = Wb @ q_int
      6) y = y_int * (beta_g * gamma / Qb)  (dequant & scale)
    STE: gradient flows to weight_fp via Wc.
    """
    def __init__(self, in_features, out_features, bias=True, bits_act=8, groups=1, per_token_act=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups  # G in the paper
        self.bits_act = bits_act
        self.per_token_act = per_token_act  # True for per-token quantization at inference
        self.weight_fp = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight_fp, a=math.sqrt(5))

    def _group_stats(self, W):
        """
        W: [out, in]
        returns alpha: [out, 1] broadcastable, beta: [out, 1] broadcastable
        We divide OUT rows into G groups (approx equal size). For each group g:
          alpha_g = (G / (n*m)) * sum_{i,j} W^{(g)}_{ij}?  -- The paper defines alpha as mean (1/(n m) sum Wij).
        Implementation:
          alpha_g = mean over rows in group (scalar), but we will make shape [out_rows_in_group, 1]
          beta_g = (G / (n*m)) * ||W^(g)||_1
        """
        out, inn = W.shape
        G = self.groups
        assert out % G == 0, "out_features must be divisible by groups"
        group_size = out // G
        alphas = torch.zeros_like(W[:, :1])
        betas = torch.zeros_like(W[:, :1])
        for g in range(G):
            start = g * group_size
            end = (g + 1) * group_size
            Wg = W[start:end]  # [group_size, in]
            # alpha_g: mean over group (scalar)
            alpha_g = Wg.mean()
            # beta_g: (G / (n*m)) * ||Wg||_1 ; here n*m for the whole W is out*in; paper's formula uses group-adjust
            beta_g = (G / (out * inn)) * Wg.abs().sum()
            alphas[start:end] = alpha_g
            betas[start:end] = beta_g
        return alphas, betas  # shapes [out,1]

    def forward(self, x):
        """
        x: training: usually Float Tensor [B, T, D_in] or [B, D_in]
        We'll treat two cases: 2D (B, D) or 3D (B, T, D). For 3D and per_token_act True, quantize per token.
        """
        W = self.weight_fp  # [out, in]
        out, inn = W.shape
        # 1) group stats
        alpha, beta = self._group_stats(W)  # each [out,1]
        # 2) center weights
        Wc = W - alpha  # broadcast alpha along in dimension
        # 3) binarize
        Wb_sign = torch.sign(Wc)
        # replace zeros with -1 per sign definition in paper (sign(x)<=0 -> -1)
        Wb_sign = torch.where(Wb_sign == 0, torch.full_like(Wb_sign, -1.0), Wb_sign)
        # STE: let gradient flow to Wc
        Wb = (Wb_sign.detach() - Wc.detach()) + Wc

        # 4) activation quantization
        # Flatten input to 2D matmul compatible shape: [*, in]
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.view(B * T, D)
            if self.per_token_act:
                # quantize per token: q_int shape (B*T, D)
                q_int, gamma = absmax_quantize_to_int(x.view(B, T, D), bits=self.bits_act, per_token=True)
                # reshape q_int to (B*T, D) and gamma to (B*T, 1)
                q_int = q_int.view(B * T, D)
                gamma = gamma.view(B * T, 1)
            else:
                q_int, gamma = absmax_quantize_to_int(x_flat, bits=self.bits_act, per_token=False)
        else:
            x_flat = x  # [B, D]
            q_int, gamma = absmax_quantize_to_int(x_flat, bits=self.bits_act, per_token=False)

        # 5) matmul: y_int = Wb @ q_int^T  (we will compute in float but q_int are integer-valued floats)
        # For efficiency do F.linear with Wb and q_int
        # q_int shape matches in_features
        y_int = F.linear(q_int, Wb, bias=None)  # [*, out]

        # 6) dequantize/rescale: y = y_int * (beta * gamma / Qb) + bias
        Qb = 2 ** (self.bits_act - 1)
        # beta is [out,1] -> [1, out] for broadcasting with y_int [*, out]
        scaling = (beta.view(1, out) * (gamma.view(-1, 1) / Qb))
        y = y_int * scaling
        if self.bias is not None:
            y = y + self.bias.view(1, out)
        # reshape back to original
        if x.dim() == 3:
            B, T, D = orig_shape
            y = y.view(B, T, out)
        return y