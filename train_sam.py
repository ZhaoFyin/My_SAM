import warnings
warnings.filterwarnings("ignore")

import argparse
from SAM.build_sam import sam_model_registry
import random
import datetime
import torch
import numpy as np
import os
from utils import trainer_synapse, Tee
from torch.utils.data import DataLoader

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

dataset_name = "vaihingen"
lora_cfg = {
    "enable": True,
    "r": 8, "alpha": 16, "dropout": 0.0,
    "target_modules": ["qkv", "proj"],
    "target_blocks": "indices",
    "indices": [8, 9, 10, 11],
    "lr_rate": 1
}


class_dict = {"BJL": ['no_landslide', 'landslide'],
              "YYL": ['no_landslide', 'landslide'],
              "uavid": ['Building', 'Road', 'Tree', 'LowVeg', 'Moving_Car', 'Static_Car', 'Human', 'Clutter'],
              "vaihingen": ['ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter'],
              "potsdam": ['ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter']}

data_dict = {'YYL': r"C:\Users\48188\Data\VOCdevkit_YYL",
             'BJL': r"C:\Users\48188\Data\VOCdevkit_BJL",
             'uavid': r"data/uavid",
             'vaihingen': r"C:\Users\48188\Desktop\Mult-scale-SAM-main\data\vaihingen_boundary",
             'potsdam': r"data/potsdam00"}

CLASSES = class_dict[dataset_name]


def main(args, snapshot_path):
    args.is_pretrain = True

    from MySamModel import ScaSAM
    net = ScaSAM(lora_cfg, num_classes=args.num_classes).cuda()

    print(f"vit可训练参数数量: {sum(p.numel() for p in net.net.image_encoder.parameters() if p.requires_grad):,}")
    print(f"总可训练参数数量: {sum(p.numel() for p in net.net.parameters() if p.requires_grad):,}")
    print(f"总参数数量: {sum(p.numel() for p in net.net.image_encoder.parameters()):,}")
    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    if dataset_name in ['YYL', 'BJL']:
        from dataset.landslide_dataset import LandslideDataset

        train_dataset = LandslideDataset(voc_root=data_dict[args.dataset], txt_name="train.txt")
        val_dataset = LandslideDataset(voc_root=data_dict[args.dataset], txt_name="val.txt")
    elif dataset_name == 'uavid':
        from dataset.uavid_dataset import UAVIDDataset, train_aug, val_aug
        train_dataset = UAVIDDataset(data_root=os.path.join(data_dict[dataset_name], 'train'),
                                     img_dir='images', mask_dir='masks',
                                     mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(1024, 1024))

        val_dataset = UAVIDDataset(data_root=os.path.join(data_dict[dataset_name], 'val'),
                                   img_dir='images', mask_dir='masks', mode='val',
                                   mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))
    elif dataset_name == 'vaihingen':
        from dataset.vaihigen_dataset import VaihingenDataset, train_aug, val_aug

        train_dataset = VaihingenDataset(data_root=os.path.join(data_dict[dataset_name], 'train'),
                                         mode='train',
                                         mosaic_ratio=0.25, transform=train_aug)

        val_dataset = VaihingenDataset(data_root=os.path.join(data_dict[dataset_name], 'test'), transform=val_aug)
    elif dataset_name == 'potsdam':
        from dataset.potsdam_dataset import PotsdamDataset, train_aug, val_aug
        train_dataset = PotsdamDataset(data_root=os.path.join(data_dict[dataset_name], 'train'),
                                       mode='train',
                                       mosaic_ratio=0.25, transform=train_aug)

        val_dataset = PotsdamDataset(data_root=os.path.join(data_dict[dataset_name], 'val'),
                                     transform=val_aug)
    else:
        raise ValueError("Dataset not supported")

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1,
                            num_workers=8,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)

    trainer_synapse(args, net, snapshot_path, train_loader, val_loader, multimask_output, CLASSES)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./results')
    parser.add_argument('--dataset', type=str,
                        default=dataset_name, help='dataset_name')
    parser.add_argument('--experiment', type=str,
                        default='MySAM', help='experiment_name')

    parser.add_argument('--num_classes', type=int,
                        default=len(CLASSES), help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=200, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.0006,
                        help='segmentation network learning rate')
    parser.add_argument('--backbone_lr', type=float, default=6e-5,
                        help='segmentation backbone network learning rate')
    parser.add_argument('--backbone_weight_decay', type=float, default=2.5e-4,
                        help='backbone weight decay')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay')
    parser.add_argument('--img_size', type=int,
                        default=512, help='input patch size of network input')
    parser.add_argument('--input_size', type=int, default=1024, help='The input size for training SAM model')
    parser.add_argument('--att', default="CBAM")
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--vit_name', type=str,
                        default='vit_b', help='select one vit model')
    parser.add_argument('--warmup', action='store_true', default=True,
                        help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=100,
                        help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--AdamW', action='store_true', default=True, help='If activated, use AdamW to finetune SAM model')
    parser.add_argument('--module', type=str, default='MySAM')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--lora_cfg', default=lora_cfg)
    args = parser.parse_args()
    now = datetime.datetime.now()
    now = now.strftime("%m%d_%H%M%S")
    snapshot_path = os.path.join(args.output, "{}".format(args.dataset), now)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    with Tee(os.path.join(snapshot_path, "running.txt"), 'w'):
        main(args, snapshot_path)
