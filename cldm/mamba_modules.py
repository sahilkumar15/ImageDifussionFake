# code/ImageDifussionFake/cldm/mamba_modules.py

"""
mamba_modules.py
================
Dual-Identity Mamba Fusion (DIMF) for DiffusionFake deepfake detection.

Novelty for ACM-MM:
  DiffusionFake disentangles source/target identity features but classifies
  using only a global average-pooled vector, discarding spatial structure.
  We introduce two complementary Mamba-inspired state-space modules:

  1. SpatialMambaClassifier (SMC):
       Treats the encoder feature map as a spatial token sequence and applies
       a selective state-space model (SSM) to capture fine-grained artifact
       patterns across spatial locations.

  2. DualIdentityMambaFusion (DIMF):
       Processes source-related and target-related features as separate token
       sequences, models their intra-sequence dynamics via a selective state-space block inspired by Mamba, then fuses
       them for identity-inconsistency-aware classification.

Both modules are pure-PyTorch (no CUDA extension required).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ────────────────────────────────────────────────────────────────────────────
# 1.  Core Mamba block (simplified selective SSM, pure PyTorch)
# ────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Simplified selective-SSM block inspired by Mamba (Gu & Dao, 2023).

    Input / output shape: (B, L, d_model)

    Key components:
      * Input projection  → inner dimension with gating (x, z split)
      * Depthwise Conv1d  → local context aggregation
      * Selective SSM     → input-dependent state transitions (dA, dB, C)
      * Gated output      → y * silu(z)
      * Residual add      → stable training
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = int(d_model * expand)
        self.d_inner = d_inner

        self.norm = nn.LayerNorm(d_model)

        # (x, z) split projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Causal depthwise conv
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=d_inner, bias=True
        )

        # Input-dependent SSM parameters: [dt(1), B(d_state), C(d_state)]
        self.x_proj = nn.Linear(d_inner, 1 + d_state + d_state, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # A matrix: fixed, diagonal (log-space for positivity)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(d_inner, -1)          # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection scalar D
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_inner)
        Returns y: (B, L, d_inner)
        """
        B, L, D = x.shape
        S = self.d_state

        x_fp = x.float()
        bcdt = self.x_proj(x_fp)                         # (B, L, 1+S+S)
        dt_raw, B_ssm, C_ssm = bcdt.split([1, S, S], dim=-1)

        dt = F.softplus(self.dt_proj(dt_raw))            # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())               # (d_inner, S)

        # Discretise via zero-order hold
        # dA: (B, L, d_inner, S)
        dA = torch.exp(dt.unsqueeze(-1) * A[None, None])
        # dB: (B, L, d_inner, S)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)

        # Sequential scan — acceptable for short sequences (49 or 196 tokens)
        h = x.new_zeros(B, D, S)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x[:, i].unsqueeze(-1)  # (B,D,S)
            # C: (B, S)  →  broadcast over D
            y_i = (h * C_ssm[:, i].unsqueeze(1)).sum(-1)           # (B, D)
            ys.append(y_i)

        y = torch.stack(ys, dim=1)                        # (B, L, D)
        y = y + x_fp * self.D[None, None]                 # skip
        return y.to(x.dtype)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        residual = x
        B, L, _ = x.shape

        x = self.norm(x)

        xz = self.in_proj(x)
        x_i, z = xz.chunk(2, dim=-1)

        xc = x_i.transpose(1, 2).contiguous()
        xc = self.conv1d(xc)[..., :L]
        xc = xc.transpose(1, 2).contiguous()
        xc = F.silu(xc)

        y = self._selective_scan(xc)
        y = y * F.silu(z)

        out = self.out_proj(y)
        return self.dropout(out) + residual


# ────────────────────────────────────────────────────────────────────────────
# 2.  SpatialMambaClassifier  (SMC)
# ────────────────────────────────────────────────────────────────────────────

class SpatialMambaClassifier(nn.Module):
    """
    Treats a spatial feature map (B, C, H, W) as an H×W token sequence
    and models artifact consistency across spatial positions with a Mamba-inspired state-space module.

    Replaces the plain GlobalAvgPool + Linear classifier head in DiffusionFake.
    """

    def __init__(self, d_model: int, num_layers: int = 2,
                 d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model, bias=False)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1, bias=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat : (B, C, H, W)
        returns logit: (B, 1)
        """
        B, C, H, W = feat.shape
        # Spatial → sequence
        tokens = feat.flatten(2).transpose(1, 2).float()  # (B, H*W, C)
        tokens = self.input_proj(tokens)

        for layer in self.layers:
            tokens = layer(tokens)

        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)                        # (B, C)
        return self.head(pooled)                           # (B, 1)


# ────────────────────────────────────────────────────────────────────────────
# 3.  DualIdentityMambaFusion  (DIMF)  ← the main novelty
# ────────────────────────────────────────────────────────────────────────────

class DualIdentityMambaFusion(nn.Module):
    """
    Dual-Identity Mamba Fusion (DIMF).

    Processes source-related and target-related spatial features as
    separate token sequences with independent Mamba encoders, then fuses
    both global representations for identity-inconsistency classification.

    Story for the paper:
      Real faces → source and target features are near-identical
      Fake faces → source and target features diverge in structured ways
      Mamba captures *how* each stream evolves spatially, and the fusion
      detects asymmetric patterns that betray manipulation.
    """

    def __init__(self, d_model: int, d_reduced: int = None,
                 num_layers: int = 2, d_state: int = 16,
                 pool_size: int = 7, dropout: float = 0.1):
        """
        d_model   : input channel dimension (C)
        d_reduced : internal dimension (default: d_model // 2)
        pool_size : spatial size to pool feature maps to before tokenising
        """
        super().__init__()
        if d_reduced is None:
            d_reduced = max(d_model // 2, 128)
        self.pool_size = pool_size

        # Separate projections for source / target streams
        self.proj_s = nn.Linear(d_model, d_reduced, bias=False)
        self.proj_t = nn.Linear(d_model, d_reduced, bias=False)

        # Independent Mamba encoders
        self.mamba_s = nn.ModuleList([
            MambaBlock(d_reduced, d_state=d_state, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.mamba_t = nn.ModuleList([
            MambaBlock(d_reduced, d_state=d_state, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm_s = nn.LayerNorm(d_reduced)
        self.norm_t = nn.LayerNorm(d_reduced)

        # Cross-stream interaction: lightweight attention
        self.cross_attn = nn.MultiheadAttention(
            d_reduced, num_heads=max(1, d_reduced // 64),
            dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_reduced)

        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(d_reduced * 2, d_reduced),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_reduced, 1)
        )

    # ------------------------------------------------------------------
    def _tokenize(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W) → tokens: (B, pool_size^2, C)"""
        if feat.shape[-2:] != (self.pool_size, self.pool_size):
            feat = F.adaptive_avg_pool2d(feat, (self.pool_size, self.pool_size))
        B, C, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2).float()     # (B, H*W, C)

    # ------------------------------------------------------------------
    def forward(self, fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        """
        fs : source-related feature map  (B, C, H, W)
        ft : target-related feature map  (B, C, H, W)
        returns logit: (B, 1)
        """
        # Tokenise both streams to (B, L, C)
        ts = self._tokenize(fs)                            # (B, L, C)
        tt = self._tokenize(ft)

        # Project to reduced dim
        ts = self.proj_s(ts)                               # (B, L, d_reduced)
        tt = self.proj_t(tt)

        # Source stream: Mamba
        for layer in self.mamba_s:
            ts = layer(ts)
        ts = self.norm_s(ts)                               # (B, L, d_reduced)

        # Target stream: Mamba
        for layer in self.mamba_t:
            tt = layer(tt)
        tt = self.norm_t(tt)                               # (B, L, d_reduced)

        # Cross-stream interaction:
        # source queries target to find inconsistencies
        ts_cross, _ = self.cross_attn(ts, tt, tt)
        ts = self.cross_norm(ts + ts_cross)

        # Global pool both streams
        gs = ts.mean(dim=1)                                # (B, d_reduced)
        gt = tt.mean(dim=1)

        # Fuse and classify
        fused = torch.cat([gs, gt], dim=-1)                # (B, 2*d_reduced)
        return self.fusion_head(fused)                     # (B, 1)


# ────────────────────────────────────────────────────────────────────────────
# 4.  MambaFakeHead  ← the unified drop-in module for diffusionfake.py
# ────────────────────────────────────────────────────────────────────────────

class MambaFakeHead(nn.Module):
    """
    Unified classification head combining three signals:
      1. logit_global  : original GlobalAvgPool + Linear (baseline)
      2. logit_spatial : SpatialMambaClassifier on backbone features
      3. logit_dual    : DualIdentityMambaFusion on source/target features

    Final logit = learnable weighted sum of the three signals.
    """

    def __init__(self, d_model: int = 1792, d_reduced: int = 512,
                 num_mamba_layers: int = 2, d_state: int = 16,
                 pool_size: int = 7, dropout: float = 0.1):
        super().__init__()

        # Baseline (replaces self.fc in GuideNet)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(d_model, 1)
        )

        # Spatial Mamba
        self.spatial_mamba = SpatialMambaClassifier(
            d_model, num_layers=num_mamba_layers,
            d_state=d_state, dropout=dropout
        )

        # Dual-identity Mamba
        self.dual_mamba = DualIdentityMambaFusion(
            d_model, d_reduced=d_reduced,
            num_layers=max(1, num_mamba_layers - 1),
            d_state=d_state, pool_size=pool_size,
            dropout=dropout
        )

        # Learnable fusion weights (initialised equally)
        self.logit_weights = nn.Parameter(torch.ones(3) / 3.0)

    # ------------------------------------------------------------------
    def forward(self, feature: torch.Tensor,
                fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        """
        feature : backbone feature map  (B, C, H, W)
        fs      : source-branch feature (B, C, H, W)  [after upsample convs]
        ft      : target-branch feature (B, C, H, W)

        returns logit: (B, 1)
        """
        w = F.softmax(self.logit_weights, dim=0)

        logit_g = self.global_head(feature)                # (B, 1)
        logit_s = self.spatial_mamba(feature)              # (B, 1)
        logit_d = self.dual_mamba(fs, ft)                  # (B, 1)

        return w[0] * logit_g + w[1] * logit_s + w[2] * logit_d