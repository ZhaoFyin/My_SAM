import torch
import torch.nn as nn
from SAM.build_sam import sam_model_registry


class ScaSAM(nn.Module):
    def __init__(self, lora_cfg, num_classes):
        super(ScaSAM, self).__init__()

        self.net, _ = sam_model_registry["vit_b"](image_size=512,
                                                  num_classes=num_classes,
                                                  checkpoint=r"E:\edgeDownload/sam_vit_b_01ec64.pth",
                                                  pixel_mean=[0.3394, 0.3598, 0.3226],
                                                  pixel_std=[0.2037, 0.1899, 0.1922],
                                                  use_sca=True,
                                                  lora_cfg=lora_cfg,
                                                  process_cp=lora_cfg["enable"],
                                                  report=True)
        for p in self.net.image_encoder.parameters():
            p.requires_grad = False
        for n, p in self.net.image_encoder.named_parameters():
            if "A" in n or "B" in n:
                p.requires_grad = True
        for n, p in self.net.image_encoder.named_parameters():
            if "sca" in n:
                p.requires_grad = True

    def forward(self, x, multi_output, img_size):
        return self.net(x, multi_output, img_size)
