# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_
import math
from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image embedding, using a transformer architecture.
        (Prompt embeddings have been completely removed.)

        Args:
            transformer_dim (int): channel dim of the transformer
            transformer (nn.Module): the transformer used to predict masks
            num_multimask_outputs (int): number of mask tokens (e.g., 3)
            activation (nn.Module): activation used in upscaling head
            iou_head_depth (int): MLP depth for IoU prediction head
            iou_head_hidden_dim (int): hidden dim for IoU head
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        structural_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image embeddings and positional encodings.
        (No prompt embeddings.)

        Args:
            image_embeddings (Tensor): [B, C, H, W]
            image_pe (Tensor):         [B, C, H, W] positional encodings
            structural_embeddings (Tensor): kept for future use (currently unused)
            multimask_output (bool): kept for interface compatibility

        Returns:
            masks (Tensor): [B, num_masks, H', W']
            iou_pred (Tensor): [B, num_masks]
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            structural_embeddings=structural_embeddings,
        )
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        structural_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks without any prompt embeddings."""
        B, C, H, W = image_embeddings.shape

        # 1) Build only output tokens (IoU + mask tokens), no prompt tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)  # [1+M, C]
        tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 1+M, C]

        # 2) Source features & positional encodings (no prompt-based repeat/add)
        src = image_embeddings  # [B, C, H, W]
        pos_src = image_pe
        if pos_src.shape[-2:] != src.shape[-2:]:
            pos_src = F.interpolate(pos_src, size=(H, W), mode="bilinear", align_corners=False)


        # 3) Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)  # hs: [B, 1+M, C]
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # 4) Upscale & hypernetwork heads
        src = src.transpose(1, 2).view(B, C, H, W)
        upscaled_embedding = self.output_upscaling(src)  # [B, C', H', W']

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, num_masks, C']

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [B, num_masks, h, w]

        # 5) IoU quality prediction
        iou_pred = self.iou_prediction_head(iou_token_out)  # [B, num_masks]

        return masks, iou_pred


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
