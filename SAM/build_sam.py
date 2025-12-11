import torch
from functools import partial
from torch.nn import functional as F
from .modeling_CNN import ImageEncoderViT, MaskDecoder, TwoWayTransformer
from .sam import Sam


def build_sam_vit_h(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


def build_sam_vit_b(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None, use_sca=True, lora_cfg=None, process_cp=False, report=False):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        use_sca=use_sca,
        lora_cfg=lora_cfg,
        process_cp=process_cp,
        report=report
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def load_with_encoder_report(
    model: torch.nn.Module,
    state_dict: dict,
    encoder_attr: str = "image_encoder",
    strict: bool = False,
):
    """
    仅针对 model.<encoder_attr>（默认 'image_encoder'）统计：
      - encoder_missing: 模型编码器里有，但 ckpt 没有的键
      - encoder_size_mismatch: 模型编码器里存在对应键，但 shape 不一致

    同时只加载 “键存在且 shape 完全一致” 的部分到整个 model（strict 可选）。
    返回 (report, loaded_subset_state_dict)
    """
    model_sd = model.state_dict()
    enc_prefix = encoder_attr + "."
    # 只取编码器范围内的模型参数键
    enc_model_keys = [k for k in model_sd.keys() if k.startswith(enc_prefix)]

    matched = {}
    encoder_size_mismatch = {}  # k -> (ckpt_shape, model_shape)
    encoder_missing = []        # keys in model encoder but not in ckpt

    # 逐项对齐（仅编码器）
    for k in enc_model_keys:
        if k in state_dict:
            if tuple(state_dict[k].shape) == tuple(model_sd[k].shape):
                matched[k] = state_dict[k]
            else:
                encoder_size_mismatch[k] = (tuple(state_dict[k].shape), tuple(model_sd[k].shape))
        else:
            encoder_missing.append(k)

    # 实际加载：只喂“匹配成功”的键（避免 size mismatch 报错）
    load_msg = model.load_state_dict(matched, strict=strict)

    # 再做一次“未加载键”的精确过滤（防御性；严格只看编码器前缀）
    # 注意：load_state_dict 的 missing_keys/unexpected_keys 是“相对传入 matched”的结果，
    # 这里我们还是以 enc_model_keys - matched 为准。
    not_loaded = set(encoder_missing) | set(encoder_size_mismatch.keys())

    # 打印简洁报告（只编码器）
    total_enc = len(enc_model_keys)
    loaded_enc = len(matched)
    print("\n===== Image Encoder Loading Report =====")
    print(f"Encoder total tensors : {total_enc}")
    print(f"Encoder loaded        : {loaded_enc}")
    print(f"Encoder NOT loaded    : {len(not_loaded)}\n")
    print(f"Encoder missing       : {sum(1 for s in encoder_missing if 'A' in s or 'B' in s)} in lora \t {sum(1 for s in encoder_missing if 'A' not in s and 'B' not in s)} in base")
    print(f"Encoder mismatch      : {sum(1 for s in encoder_size_mismatch if 'A' in s or 'B' in s)} in lora \t {sum(1 for s in encoder_size_mismatch if 'A' not in s and 'B' not in s)} in base")
    print("=======================================\n")


def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        num_classes,
        image_size,
        pixel_mean,
        pixel_std,
        use_sca=True,
        checkpoint=None,
        lora_cfg=None,
        process_cp=False,
        report=False
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            use_sca=use_sca,
            lora_cfg=lora_cfg
        ),
        mask_decoder=MaskDecoder(
            # num_multimask_outputs=3,
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )
    # sam.eval()
    sam.train()
    if checkpoint is not None:
        state_dict = torch.load(checkpoint)

        if process_cp:
            state_dict = remap_ckpt_for_lora(state_dict=state_dict)
        if report:
            load_with_encoder_report(sam, state_dict)

        try:
            sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
            sam.load_state_dict(new_state_dict)

    return sam, image_embedding_size


def load_from(sam, state_dict, image_size, vit_patch_size):
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict


def remap_ckpt_for_lora(state_dict: dict) -> dict:
    for k in list(state_dict.keys()):
        parts = k.split(".")
        if len(parts) < 2:
            # 没有“倒数第二段”的结构，跳过
            continue
        parts.insert(-1, "base")
        new_k = ".".join(parts)
        # 若目标键不存在，则新增一份
        if new_k not in state_dict:
            state_dict[new_k] = state_dict[k]
    return state_dict


