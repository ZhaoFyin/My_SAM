# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic
from typing import Any, Dict, List, Tuple
import torchvision
from SAM.modeling_CNN import ImageEncoderViT, MaskDecoder


def build_2d_sincos_position_embedding(h: int, w: int, embed_dim: int,
                                       device=None, dtype=None, temperature: float = 10000.):
    """Return [1, C, H, W] 2D sin-cos PE. C should be divisible by 4; if not, we pad."""
    if device is None: device = torch.device('cpu')
    if dtype  is None: dtype  = torch.float32

    grid_h = torch.arange(h, dtype=torch.float32, device=device)
    grid_w = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(grid_h, grid_w, indexing='ij')  # [H, W]

    dim_half = embed_dim // 2
    dim_quarter = embed_dim // 4
    if dim_quarter == 0:
        raise ValueError(f"embed_dim must be >= 4; got {embed_dim}")

    omega = torch.arange(dim_quarter, device=device, dtype=torch.float32) / dim_quarter
    omega = 1.0 / (temperature ** omega)  # [C/4]

    out_y = torch.einsum('hw,c->hwc', yy, omega)  # [H, W, C/4]
    out_x = torch.einsum('hw,c->hwc', xx, omega)  # [H, W, C/4]

    pos = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=2)  # [H,W,C]
    pos = pos.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)  # [1,C,H,W]

    # 若 embed_dim 不是 4 的倍数，做 0 填充
    if pos.shape[1] < embed_dim:
        pad_c = embed_dim - pos.shape[1]
        pos = F.pad(pos, (0,0,0,0,0,pad_c))  # pad channel dim

    return pos  # [1, C, H, W]


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
    
    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def get_dense_pe(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        # image_embeddings: [B, C, H, W]
        B, C, H, W = image_embeddings.shape
        pe = build_2d_sincos_position_embedding(
            h=H, w=W, embed_dim=C,
            device=image_embeddings.device,
            dtype=image_embeddings.dtype
        )  # [1, C, H, W]
        return pe.expand(B, -1, -1, -1)  # [B, C, H, W]

    def forward(self, batched_input, multimask_output, image_size):
        outputs = self.forward_train(batched_input, multimask_output, image_size)
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size):
        # image_size = batched_input.shape[-1] # if test
        image_embeddings = self.image_encoder(batched_input)
        image_pe = self.get_dense_pe(image_embeddings)
        fusion_features = None

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            structural_embeddings=fusion_features,
            multimask_output=multimask_output
        )
        
        masks = self.postprocess_masks(
                low_res_masks,
                input_size=(image_size, image_size),
                original_size=(image_size, image_size)
            )
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }
        return outputs
        # return masks # if test

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

       


        
