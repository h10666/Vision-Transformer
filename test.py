# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist
import cv2

from tqdm import tqdm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

# from models.modeling import VisionTransformer, CONFIGS, complete_model
# from models.vit_tokenlearner import ViT, complete_model
# from models.modeling_fusion import VisionTransformer, CONFIGS, complete_model
from models.modeling_tl import VisionTransformer, CONFIGS, complete_model
# from models.modeling_early import VisionTransformer, CONFIGS
# from models.modeling_mbt import fusion_model, CONFIGS

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

from thop import profile, clever_format
import pandas as pd
import matplotlib.pyplot as mp
import seaborn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def simple_accuracy(preds, labels):
    print(preds)
    print(labels)
    return (preds == labels).mean()


def Accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).to(torch.float32).sum(0)
        res.append(float(correct_k.mul_(100.0 / batch_size)))
    return tuple(res)


def main():
    parser = argparse.ArgumentParser(description="Standard video-level" +
                                                 " testing")
    parser.add_argument('--dataset', type=str,
                        choices=['ucf101', 'hmdb51', 'kinetics', 'epic-kitchens-55', 'epic-kitchens-100', 'vggsound'])
    parser.add_argument('--modality', type=str,
                        choices=['RGB', 'Flow', 'RGBDiff', 'Spec'],
                        nargs='+', default=['RGB', 'Flow', 'Spec'])
    parser.add_argument('--checkpoint_name', type=str)
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument('--train_list', type=str)
    parser.add_argument('--val_list', type=str)
    parser.add_argument('--visual_path', type=str, default="")
    parser.add_argument('--audio_path', type=str, default="")
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--resampling_rate', type=int, default=24000)
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Total batch size for training.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stochastic_droplayer_rate', type=float, default=0.0)
    parser.add_argument('--shift_div', type=int, default=4)



    global args
    args = parser.parse_args()
    # Set seed
    set_seed(args)
    config = CONFIGS[args.model_type]

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'beoid':
        num_class = 34
    elif args.dataset == 'epic-kitchens-55':
        num_class = (125, 352)
    elif args.dataset == 'epic-kitchens-100':
        num_class = (97, 300)
    elif args.dataset == 'vggsound':
        num_class = 10
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    config.n_seg = args.num_segments
    config.fold_div = args.shift_div
    config.stochastic_droplayer_rate = args.stochastic_droplayer_rate

    print(torch.cuda.is_available())
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_losses = AverageMeter()

    # model = VisionTransformer(config, zero_head=True, num_classes=num_class)
    model = complete_model(config, num_classes=num_class)
    # model = fusion_model(config, num_classes=num_class)


    model.to(args.device)

    weights = './output/{checkpoint_name}'.format(checkpoint_name=args.checkpoint_name)
    checkpoint = torch.load(weights)
    base_dict = {k: v for k, v in list(checkpoint.items())}
    model.load_state_dict(base_dict)
    train_loader, test_loader = get_loader(args)
    model.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        x, y, mask = batch
        # print(x.shape, y.shape)
        for m in args.modality:
            x[m] = x[m].to(args.device)
        y = y.to(args.device)
        batch_size = x['Spec'].size(0)
        with torch.no_grad():
            logits = model(x, labels=y)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            # preds = torch.argmax(logits, dim=-1)

        # if len(all_preds) == 0:
        #     all_preds.append(preds.detach().cpu().numpy())
        #     all_label.append(y.detach().cpu().numpy())
        # else:
        #     all_preds[0] = np.append(
        #         all_preds[0], preds.detach().cpu().numpy(), axis=0
        #     )
        #     all_label[0] = np.append(
        #         all_label[0], y.detach().cpu().numpy(), axis=0
        #     )

            prec1, prec5 = Accuracy(logits, y, topk=(1, 5))

        top1.update(prec1, batch_size)
        top5.update(prec5, batch_size)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    # all_preds, all_label = all_preds[0], all_label[0]
    # accuracy = simple_accuracy(all_preds, all_label)

    print("Valid Accuracy-1: %2.5f" % top1.avg)
    print("Valid Accuracy-5: %2.5f" % top5.avg)

    model1 = complete_model(config, num_classes=num_class)
    inputs_video = torch.randn(1, 24, 224, 224)
    inputs_audio = torch.randn(1, 8, 128, 128)
    inputs = {'RGB': inputs_video, 'Spec': inputs_audio}
    labels = torch.randn(1, 1)

    macs, params = profile(model1, inputs=(inputs, labels))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)


if __name__ == '__main__':
    main()
