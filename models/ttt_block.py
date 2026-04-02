# --------------------------------------------------------
# ViT^3: Unlocking Test-Time Training in Vision
# Written by Dongchen Han
# --------------------------------------------------------

import torch
import torch.nn as nn
from .swin import trunc_normal_
import torch.nn.functional as F


class TTT(nn.Module):
    r""" Test-Time Training block for ViT^3 model.
        - https://arxiv.org/abs/2512.01643

    This block implements test-time inner training of two parallel sub-modules:
        1. Simplified SwiGLU inner module, i.e., SwiGLU with identity output layer
        2. 3x3 depth-wise convolution (3x3dwc) inner module

    Note:
        The TTT inner loss is a per-head / per-sample vector-valued loss (shape [B, num_heads]).
        The torch.autograd.backward only supports scalar losses, so here we implement a hand-derived
        backward (closed-form gradient expressions) that directly computes parameter gradients.
        Alternative efficient implementations are welcome and appreciated.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        use_conv_branch (bool, optional): If True, include depthwise conv branch. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, use_conv_branch=True, **kwargs):

        super().__init__()
        head_dim = dim // (num_heads)
        self.dim = dim
        self.num_heads = (num_heads)
        self.use_conv_branch = use_conv_branch

        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        if self.use_conv_branch:
            self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
            self.proj = nn.Linear(dim + head_dim, dim)
        else:
            self.proj = nn.Linear(dim, dim)
        
        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)
        if self.use_conv_branch:
            trunc_normal_(self.w3, std=.02)

        equivalent_head_dim = 9
        self.scale = equivalent_head_dim ** -0.5
        # The equivalent head_dim of 3x3dwc branch is 1x(3x3)=9 (1 channel, 3x3 kernel)
        # We used this equivalent_head_dim to compute self.scale in our earlier experiments
        # Using self.scale=head_dim**-0.5 (head_dim of simplified SwiGLU branch) leads to similar performance

    def inner_train_simplified_swiglu(self, k, v, w1, w2, lr=1.0):
        """
        Args:
            k (torch.Tensor): Key tensor of shape [B, num_heads, N, head_dim]
            v (torch.Tensor): Value tensor of shape [B, num_heads, N, head_dim]
            w1 (torch.Tensor): First weight matrix of shape [1, num_heads, head_dim, head_dim]
            w2 (torch.Tensor): Second weight matrix of shape [1, num_heads, head_dim, head_dim]
            lr (float, optional): Learning rate for inner-loop update. Default: 1.0

        Returns:
            tuple: Updated w1 and w2
        """
        # --- Forward ---
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig
        # v_hat = a
        # l = (v_hat * v).sum(dim=3).mean(dim=2) * self.scale
        # Notably, v_hat and l are not computed here because
        # they are unnecessary for deriving the gradient expression below.
        # We directly compute e = dl/dv_hat for the backward pass.

        # --- Backward ---
        e = - v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        # --- Clip gradient (for stability) ---
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        # --- Step ---
        w1, w2 = w1 - lr * g1, w2 - lr * g2
        return w1, w2

    def inner_train_3x3dwc(self, k, v, w, lr=1.0, implementation='prod'):
        """
        Args:
            k (torch.Tensor): Spatial key tensor of shape [B, C, H, W]
            v (torch.Tensor): Spatial value tensor of shape [B, C, H, W]
            w (torch.Tensor): 3x3 convolution weights of shape [C, 1, 3, 3]
            lr (float, optional): Learning rate for inner-loop update. Default: 1.0
            implementation (str, optional): Implementation method, 'conv' or 'prod'. Default: 'prod'

        Returns:
            torch.Tensor: Updated convolution weights
        """
        # --- Forward ---
        # v_hat = F.conv2d(k, w, padding=1, groups=C)
        # l = - (v_hat * v).mean(dim=[-2, -1]) * self.scale
        # Notably, v_hat and l are not computed here because
        # they are unnecessary for deriving the gradient expression below.
        # We directly compute e = dl/dv_hat for the backward pass.

        # --- Backward ---
        # Two equivalent implementations. The 'prod' implementation appears to be slightly faster
        B, C, H, W = k.shape
        e = - v / float(v.shape[2] * v.shape[3]) * self.scale
        if implementation == 'conv':
            g = F.conv2d(k.reshape(1, B * C, H, W), e.reshape(B * C, 1, H, W), padding=1, groups=B * C)
            g = g.transpose(0, 1)
        elif implementation == 'prod':
            k = F.pad(k, (1, 1, 1, 1))
            outs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys = 1 + dy
                    xs = 1 + dx
                    dot = (k[:, :, ys: ys + H, xs: xs + W] * e).sum(dim=(-2, -1))
                    outs.append(dot)
            g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        else:
            raise NotImplementedError

        # --- Clip gradient (for stability) ---
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)

        # --- Step ---
        w = w.repeat(B, 1, 1, 1) - lr * g
        return w

    def forward(self, x, h, w, rope=None):
        """
        Args:
            x (torch.Tensor): Input features with shape of (B, N, C)
            h (int): Feature map height
            w (int): Feature map width
            rope (nn.Module, optional): Rotary Position Embedding
        """
        b, n, c = x.shape
        d = c // self.num_heads

        # Prepare q/k/v
        q1, k1, v1, q2, k2, v2 = torch.split(self.qkv(x), [c, c, c, d, d, d], dim=-1)
        if rope is not None:
            q1 = rope(q1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
            k1 = rope(k1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
        else:
            q1 = q1.reshape(b, n, self.num_heads, d).transpose(1, 2)
            k1 = k1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        v1 = v1.reshape(b, n, self.num_heads, d).transpose(1, 2)

        # Inner training using (k, v)
        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)

        # Apply updated inner module to q
        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(b, n, c)

        if self.use_conv_branch:
            q2 = q2.reshape(b, h, w, d).permute(0, 3, 1, 2)
            k2 = k2.reshape(b, h, w, d).permute(0, 3, 1, 2)
            v2 = v2.reshape(b, h, w, d).permute(0, 3, 1, 2)
            w3 = self.inner_train_3x3dwc(k2, v2, self.w3, implementation='prod')
            x2 = F.conv2d(q2.reshape(1, b * d, h, w), w3, padding=1, groups=b * d)
            x2 = x2.reshape(b, d, n).transpose(1, 2)
            x = torch.cat([x1, x2], dim=-1)
        else:
            x = x1

        # Output proj
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
    
    def flops(self, B: int, N: int, h: int, w: int, implementation: str = "prod") -> int:
        """
        MAC-count (same convention as Swin's flops): 1 MAC = 1.
        Counts major matmuls/convs in forward(). Ignores elementwise ops.

        Args:
            B: batch size *entering this module*
               (for Swin-window usage, this is B*nW because windows are packed in batch dim)
            N: number of tokens (should equal h*w)
            h,w: spatial dims
            implementation: 'prod' or 'conv' for inner_train_3x3dwc (same MAC scale)
        """
        C = self.dim
        Hh = self.num_heads
        assert C % Hh == 0, "dim must be divisible by num_heads"
        d = C // Hh
        assert N == h * w, f"N must equal h*w, got N={N}, h*w={h*w}"

        flops = 0

        # 1) qkv projection: (B, N, C) -> (B, N, 3C + 3d)
        flops += B * N * C * (3 * C + 3 * d)

        # 2) inner_train_simplified_swiglu:
        # z1 = k @ w1, z2 = k @ w2
        flops += 2 * (B * Hh * N * d * d)
        # g1 = k^T @ (...), g2 = k^T @ (...)
        flops += 2 * (B * Hh * N * d * d)

        # 3) apply updated module to q: (q @ w1) and (q @ w2)
        flops += 2 * (B * Hh * N * d * d)

        if self.use_conv_branch:
            # 4) inner_train_3x3dwc: ~9 taps per spatial location per channel
            flops += 9 * B * d * h * w
            # 5) depthwise 3x3 conv apply on q2
            flops += 9 * B * d * h * w
            # 6) output projection: (C + d) -> C
            flops += B * N * (C + d) * C
        else:
            # 4) output projection: C -> C
            flops += B * N * C * C

        return int(flops)

    def flops_per_window_per_sample(self, window_size: int) -> int:
        """
        Convenience: MACs for ONE window and ONE sample (B=1).
        Useful for Swin-style flops() which ignores batch.
        """
        Ws = window_size
        N = Ws * Ws
        return self.flops(B=1, N=N, h=Ws, w=Ws)