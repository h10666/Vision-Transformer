import typing
import io
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms
from utils.data_utils import get_loader

from models.modeling import VisionTransformer, CONFIGS
from models.modeling_tl import VisionTransformer, complete_model, CONFIGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare Model
parser = argparse.ArgumentParser(description="Standard video-level" +" testing")
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
config = CONFIGS[args.model_type]

config.n_seg = args.num_segments
config.fold_div = args.shift_div
config.stochastic_droplayer_rate = args.stochastic_droplayer_rate

print(torch.cuda.is_available())
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = VisionTransformer(config, zero_head=True, num_classes=10)
model = complete_model(config, num_classes=10)

model.to(args.device)

weights = './output/{checkpoint_name}'.format(checkpoint_name=args.checkpoint_name)
checkpoint = torch.load(weights)
base_dict = {k: v for k, v in list(checkpoint.items())}
model.load_state_dict(base_dict)
train_loader, test_loader = get_loader(args)

image_dir = "/opt/data/private/datasets/vggsound/v100_frames/P25733/00125.jpg"
im = Image.open(image_dir)
im.show()

input = []
label = []
for step, batch in enumerate(test_loader):
    x, y = batch
    for m in args.modality:
        x[m] = x[m].to(args.device)
    y = y.to(args.device)
    input.append(x)
    label.append(y)

logits, att_mat = model(input[86], labels=label[86])

att_mat = torch.stack(att_mat).squeeze(1)
att_mat = att_mat.cpu()

# Average the attention weights across all heads.
att_mat = torch.mean(att_mat, dim=1)

# To account for residual connections, we add an identity matrix to the
# attention matrix and re-normalize the weights.
residual_att = torch.eye(att_mat.size(1))
aug_att_mat = att_mat + residual_att
aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

# Recursively multiply the weight matrices
joint_attentions = torch.zeros(aug_att_mat.size())
joint_attentions[0] = aug_att_mat[0]

for n in range(1, aug_att_mat.size(0)):
    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
print(joint_attentions.shape)

# Attention from the output token to the input space.
v = joint_attentions[-1]
print('v', v.shape)
grid_size = int(np.sqrt(aug_att_mat.size(-1)))
print(grid_size)
mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
print(v[0, 1:].shape)
mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
result = (mask * im).astype("uint8")

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

ax1.set_title('Original')
ax2.set_title('Attention Map')
ax1.imshow(im)
ax2.imshow(result)
fig.savefig('4.png')




