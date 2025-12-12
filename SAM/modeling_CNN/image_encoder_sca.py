# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type, Any
# from .common import LayerNorm2d, MLPBlock
from SAM.modeling_CNN.common import LayerNorm2d, MLPBlock

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.A = nn.Linear(in_f, r, bias=False)
        self.B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.dropout = nn.Dropout(dropout)

        # 打标记，便于只优化 LoRA 参数
        for p in self.A.parameters():
            p.is_lora_param = True
        for p in self.B.parameters():
            p.is_lora_param = True

        # 冻结 base
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.dropout(self.B(self.A(x))) * self.scaling


def wrap_lora(linear: nn.Linear,
              lora_cfg: Optional[dict],
              module_name: Optional[str] = None,
              block_idx: Optional[int] = None):
    if not (lora_cfg and lora_cfg.get("enable", False)):
        return linear

        # 1) 模块过滤
    target_modules = lora_cfg.get("target_modules", None)
    if target_modules is not None and module_name is not None:
        if module_name not in target_modules:
            return linear

    # 2) 层索引过滤
    tb = lora_cfg.get("target_blocks", None)
    if tb is not None and block_idx is not None:
        if tb == "last_k":
            k = int(lora_cfg.get("last_k", 0))
            # 只有调用方保证传入 depth 才能更精确；这里用 block_idx 是否 >= depth-k 的外部约定来控制
            # 建议：在构造 Block 时，额外将 depth 也传入，或由调用方预先算好 indices 放进 lora_cfg["indices"]
            # 这里保守返回 linear；具体选择逻辑在调用方处理更稳妥
            pass
        elif tb == "indices":
            idx_list = set(lora_cfg.get("indices", []))
            if block_idx not in idx_list:
                return linear

    # 3) 真的要加 LoRA
    r = int(lora_cfg.get("r", 8))
    alpha = int(lora_cfg.get("alpha", 16))
    dropout = float(lora_cfg.get("dropout", 0.0))
    return LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        use_sca=True,
        global_attn_indexes: Tuple[int, ...] = (),
        lora_cfg: Optional[dict] = None,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.out_indices = [3, 6, 9]
        self.lora_cfg = lora_cfg or {"enable": False}

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_sca=use_sca,
                lora_cfg=self.lora_cfg,
                block_idx=i
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> tuple[Any, list[Any]]:
        x = self.patch_embed(x)
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        if self.pos_embed is not None:
            # print("self.pos_embed", self.pos_embed.shape)
            if self.pos_embed.size(2) != x.size(2):
                pos_embed = self.pos_embed.permute(0, 3, 1, 2)
                pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic')
                pos_embed = pos_embed.permute(0, 2, 3, 1)
            else:
                pos_embed = self.pos_embed
            x = x + pos_embed

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        use_sca=True,
        lora_cfg: Optional[dict] = None,
        block_idx: int = -1
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            lora_cfg=lora_cfg,
            block_idx=block_idx,
        )

        self.sca_layer = [2, 5, 8, 11]
        self.use_sca = use_sca
        self.block_idx = block_idx
        if self.block_idx in self.sca_layer and self.use_sca:
            self.norm_sca = nn.LayerNorm(dim)
            self.sca = ScaleContextAttention(dim, window_size=14, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        if self.block_idx in self.sca_layer and self.use_sca:
            x = x + self.sca(self.norm_sca(x))
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        lora_cfg: Optional[dict] = None,
        block_idx: int = -1
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        proj = nn.Linear(dim, dim)
        self.qkv  = wrap_lora(qkv,  lora_cfg, module_name="qkv",  block_idx=block_idx)
        self.proj = wrap_lora(proj, lora_cfg, module_name="proj", block_idx=block_idx)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class ScaleContextAttention(nn.Module):
    def __init__(self, dim, window_size=7, pool_scales=(2, 4), num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pool_scales = pool_scales
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # local branch QKV
        self.qkv_local = nn.Linear(dim, dim * 3, bias=False)
        self.proj_local = nn.Linear(dim, dim)

        # global-scale branch QKV
        self.qkv_global = nn.Linear(dim, dim * 3, bias=False)
        self.proj_global = nn.Linear(dim, dim)

        # 通道门控
        # 通道门控
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(self.dim // 4, 8)  # 中间维度，随便你调，防止太小
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, hidden, 1, bias=False),  # 2C -> hidden
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.dim * 2, 1, bias=True),  # hidden -> 2C
            nn.Sigmoid()
        )
        # self.gate_pool = nn.AdaptiveAvgPool2d(1)
        # self.gate_mlp = nn.Sequential(
        #     nn.Conv2d(dim * 2, dim // 4, 1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim // 4, 2, 1, bias=True),  # 输出两个通道：local / global
        #     nn.Sigmoid()
        # )

    def forward(self, x):

        B, H, W, C = x.shape
        N = H * W

        feat = x

        # ========= Local branch: window self-attn =========
        # partition windows
        windows, shape_info = window_partition(feat, self.window_size)  # [Bn, Ws*Ws, C]
        windows = windows.reshape(windows.shape[0], -1, windows.shape[-1])
        Bn, Lw, _ = windows.shape

        qkv = self.qkv_local(windows).reshape(Bn, Lw, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3,Bn,Hd,Lw,Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [Bn, heads, Lw, Dh]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out_local = (attn @ v).transpose(1, 2).reshape(Bn, Lw, C)
        out_local = self.proj_local(out_local)  # [Bn, Lw, C]

        # unpartition back
        out_local = out_local.view(Bn, Lw, C)
        out_local = out_local.view(Bn, self.window_size * self.window_size, C)
        out_local = window_unpartition(out_local, self.window_size, shape_info, (H, W))  # [B,C,H,W]

        # ========= Global-scale branch: pooled cross-attn =========
        # flatten as tokens
        tokens = feat.reshape(feat.shape[0], -1, feat.shape[-1])  # [B,N,C]
        qkv_g = self.qkv_global(tokens).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv_g = qkv_g.permute(2, 0, 3, 1, 4)
        qg, kg, vg = qkv_g[0], qkv_g[1], qkv_g[2]  # [B,heads,N,Dh]

        # 多尺度池化，作为 K/V 的补充
        kv_list = [kg]
        vv_list = [vg]
        for s in self.pool_scales:

            pooled = nn.functional.avg_pool2d(feat.permute(0, 3, 1, 2), kernel_size=s, stride=s) + \
                     nn.functional.max_pool2d(feat.permute(0, 3, 1, 2), kernel_size=s, stride=s)
            # 上采样回 H,W
            pooled = nn.functional.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            pooled_tok = pooled.flatten(2).transpose(1, 2)  # [B,N,C]
            kps = pooled_tok.unsqueeze(2)  # [B,N,1,C]
            kps = kps.reshape(B, self.num_heads, N, C // self.num_heads)
            kv_list.append(kps)
            vv_list.append(kps)  # 这里偷懒，用同一个作为 V，你也可以再单独线性映射

        K_all = torch.cat(kv_list, dim=2)  # [B,heads,N*(1+len(scales)),Dh]
        V_all = torch.cat(vv_list, dim=2)

        attn_g = (qg @ K_all.transpose(-2, -1)) * self.scale
        attn_g = attn_g.softmax(dim=-1)
        out_global = (attn_g @ V_all).transpose(1, 2).reshape(B, N, C)
        out_global = self.proj_global(out_global)
        out_global = out_global.transpose(1, 2).view(B, C, H, W)

        # ========= 3) 门控融合 =========
        # cat = torch.cat([out_local.permute(0, 3, 1, 2), out_global], dim=1)  # [B,2C,H,W]
        # gate = self.gate_mlp(self.gate_pool(cat))        # [B,2,1,1]
        # w_local = gate[:, 0:1, :, :]
        # w_global = gate[:, 1:2, :, :]
        #
        # fused = w_local * out_local + w_global * out_global.permute(0, 2, 3, 1)  # [B,C,H,W]

        out_local_chw = out_local.permute(0, 3, 1, 2)  # [B, C, H, W]
        cat = torch.cat([out_local_chw, out_global], dim=1)  # [B, 2C, H, W]
        gate = self.gate_mlp(self.gate_pool(cat))  # [B, 2C, 1, 1]

        # 按通道一分为二：前 C 个是 local 的权重，后 C 个是 global 的权重
        w_local, w_global = gate.chunk(2, dim=1)  # 各 [B, C, 1, 1]

        # 通道级融合
        fused_chw = w_local * out_local_chw + w_global * out_global  # [B, C, H, W]

        # 再转回 BHWC 供后续 Block 使用
        fused = fused_chw.permute(0, 2, 3, 1)  # [B, H, W, C]
        return fused


if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    model = ImageEncoderViT(window_size=14)
    y = model(x)
    print(y.shape)
