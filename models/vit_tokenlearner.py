"""
Refered to
 - https://arxiv.org/pdf/2106.11297.pdf
 - https://github.com/lucidrains/vit-pytorch
 - https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner

some type of size
 - image_size: width or height of the original image, like `256`
 - patch_size: patch width/height, like 16
 - patch_num: number of patch, calculated by (image_size // patch_size) ** 2, the original `sequence length` at the same time.
 - hidden_size: feature dimension, or the dimension of the last hidden state in huggingface's transformers
 - num_token: the compressed sequence length.
 - featmap_size: int(patch_num ** 0.5)

"""
from dataclasses import dataclass
from typing import Optional, List

import torch
import numpy as np
from torch import nn

from einops import rearrange


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


@dataclass()
class VitOutputs:
    feature_map: Optional[torch.Tensor] = None
    feature: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None
    last_hidden_state: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TokenLearnerModule(nn.Module):
    def __init__(self, featmap_size, hidden_size, num_tokens: int = 8, sum_pooling: bool = True, v11=False, dropout=0.,
                 *args, **kwargs):
        super().__init__()
        self.sum_pooling = sum_pooling
        self.v11 = v11

        self.ln = nn.LayerNorm(featmap_size, featmap_size)  # [bs, hidden_size, img/patch, img/patch]

        if self.v11:
            self.convs = nn.Sequential(
                *[
                    nn.Conv2d(hidden_size, hidden_size,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=0,  # = SAME
                              groups=8,
                              bias=False) for _ in range(3)
                ],
                nn.Conv2d(hidden_size, num_tokens,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding=0,  # = SAME
                          bias=False),
            )
            self.conv2 = nn.Conv2d(hidden_size, hidden_size,
                                   kernel_size=(1, 1), stride=(1, 1),
                                   padding=0,  # = SAME
                                   groups=8,  # feature_group_count
                                   bias=False)
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.convs = nn.Sequential(
                *[
                    nn.Conv2d(hidden_size, hidden_size,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=1,  # = SAME, (3-1)/2 = 1
                              bias=False) for _ in range(3)
                ],
                nn.GELU(),
                nn.Conv2d(hidden_size, num_tokens,
                          kernel_size=(3, 3), stride=(1, 1),
                          padding=1,  # = SAME, (3-1)/2 = 1
                          bias=False)
            )

    def forward(self, x):
        """

        :param x: [bs, length, hidden_size] = [bs, w*h, hidden_size]
        :return:
        """
        b, length, hidden_size = x.shape
        w = h = int((length // 8) ** 0.5)
        x = x.transpose(1, 2).reshape(b * 8, hidden_size, w, h)  # [b, hidden_size, w, h]

        selected = self.ln(x)  # [bs, hidden_size, h, w].

        selected = self.convs(selected)  # [bs, n_token, h, w].
        n_tokens = selected.shape[1]

        if self.v11:
            selected = torch.reshape(selected, (b * 8, -1, h * w))  # [bs, n_token, h*w].
            selected = torch.softmax(selected, dim=-1)

            feature = self.conv2(x)  # [bs, hidden_size, h, w].
            feature = torch.reshape(feature, (b * 8, -1, h * w))  # [bs, n_token, h*w]
            feature = feature.transpose(1, 2)  # [bs, h*w, hidden_size]

            feature = torch.bmm(selected, feature)  # einsum('...si,...id->...sd')
            feature = self.dropout(feature)  # [bs, n_token, hidden_size]
            feature = torch.reshape(feature, (b, 8 * n_tokens, -1))
        else:
            selected = torch.reshape(selected, (b, -1, h * w))  # [bs, n_token, h*w]
            selected = torch.sigmoid(selected).unsqueeze(-1)  # [bs, n_token, h*w, 1]

            feature = torch.reshape(x, (b, 1, -1, h * w))  # [bs, 1, hidden_size, h*w]
            feature = feature.transpose(2, 3)  # [bs, 1, h*w, hidden_size]

            feature = feature * selected
            if self.sum_pooling:
                feature = feature.sum(dim=-1)
            else:
                feature = feature.mean(dim=-1)

        return feature


class Transpose(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.axis = a, b

    def forward(self, x):
        return x.transpose(*self.axis)


class TokenFuser(nn.Module):
    def __init__(self, featmap_size, num_toknes, hidden_size, norm: bool = True, dropout: float = 0):
        super().__init__()
        self.for_input = nn.Sequential(
            nn.LayerNorm(hidden_size) if norm else nn.Identity(),
            Transpose(1, 2),
            nn.Linear(num_toknes, num_toknes),
            Transpose(1, 2),
            nn.LayerNorm(hidden_size) if norm else nn.Identity(),
        )

        self.for_ori = nn.Sequential(
            nn.LayerNorm(featmap_size, featmap_size),
            nn.Conv2d(hidden_size, num_toknes,
                      kernel_size=(1, 1), stride=(1, 1),
                      padding=0,  # = SAME
                      bias=False),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, token, ori):
        """

        :param token: Inputs of shape `[b, n_token, hidden_size]`.
        :param ori: Inputs of shape `[bs, length(w*h), hidden_size]`.
        :return:
        """
        b, n_tokens, hidden_size = token.shape
        token = token.reshape(b*8, n_tokens // 8, hidden_size)
        token = self.for_input(token)  # [b, n_token, hidden_size] -> [b, n_token, hidden_size]

        b, seqlen, hidden_size = ori.shape
        w = h = int((seqlen // 8) ** 0.5)
        ori = (
            ori
                .transpose(1, 2)  # [bs, hidden_size, length]
                .reshape(b*8, hidden_size, w, h)  # [bs, hidden_size, w, h]
        )
        mix = self.for_ori(ori)  # [b, length, w,h] -> [b, n_token, w,h]
        mix = mix.unsqueeze(-1)  # [bs, n_token, w, h, 1]

        token = token[:, :, None, None, :]  # [bs, n_token, 1, 1, hidden_size]

        token = token * mix  # [bs, n_token, w, h, hidden_size]
        token = token.sum(dim=1)  # [bs, w, h, hidden_size]
        token = self.dropout(token)
        token = token.reshape(b, -1, hidden_size)
        return token


class Transformer(nn.Module):
    def __init__(self, hidden_size, depth, heads, dim_head, mlp_dim,
                 tokenlearner: TokenLearnerModule,
                 tokenlearner_loc: int,
                 fuse: TokenFuser = None,
                 dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.tokenlearner = tokenlearner
        self.fuse = fuse
        self.tokenlearner_loc = tokenlearner_loc
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(hidden_size, Attention(hidden_size, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(hidden_size, FeedForward(hidden_size, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x):
        """

        :param x: [b, hidden_size, length]
        :return:
        """
        x = x.transpose(1, 2)  # b, length, hidden_size
        for i, (attn, ff) in enumerate(self.layers):
            # dynamic -> tokenlearner
            ori = x
            if i == self.tokenlearner_loc:
                # [b, length(w*h), hidden_size] -> [b, n_token, hidden_size]
                x = self.tokenlearner(x)

            # [b, length/n_token, hidden_size] -> [b, length/n_token, hidden_size]
            x = attn(x) + x
            x = ff(x) + x

            if i == self.tokenlearner_loc and self.fuse is not None:
                # [b, n_token, hidden_size] -> [b, length, hidden_size]
                x = self.fuse(x, ori)
                x = x + ori

        return x


class ViT(nn.Module):

    def __init__(self, *,
                 image_size,
                 patch_size,
                 hidden_size,
                 num_tokens,
                 num_classes,
                 in_channels,
                 fuse,
                 v11,
                 tokenlearner_loc,
                 depth, heads, mlp_dim, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0., fusion=False):
        """

        :param image_size:
        :param patch_size:
        :param hidden_size:
        :param num_tokens:
        :param fuse: Whether to use TokenFuser as well.
        :param v11: see original implementaion for details.
        :param tokenlearner_loc: The layer indices to add TokenLearner to.
        :param num_classes:
        :param in_channels:
        :param depth:
        :param heads:
        :param mlp_dim:
        :param pool:
        :param dim_head:
        :param dropout:
        :param emb_dropout:
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.feature_dim = hidden_size
        self.embedding = nn.Conv2d(in_channels, hidden_size, (patch_size, patch_size), (patch_size, patch_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, hidden_size, 8 * (image_size // patch_size) ** 2))

        self.dropout = nn.Dropout(emb_dropout)

        featmap_size = image_size // patch_size

        fuse_module = None
        if fuse:
            fuse_module = TokenFuser(featmap_size=featmap_size,
                                     num_toknes=num_tokens,
                                     hidden_size=hidden_size,
                                     norm=True,
                                     dropout=dropout)

        tokenlearner = TokenLearnerModule(featmap_size=featmap_size,
                                          hidden_size=hidden_size,
                                          num_tokens=num_tokens,
                                          v11=v11,
                                          sum_pooling=True,
                                          dropout=dropout)

        self.transformer = Transformer(hidden_size=hidden_size, depth=depth, heads=heads, dim_head=dim_head,
                                       mlp_dim=mlp_dim,
                                       tokenlearner=tokenlearner,
                                       tokenlearner_loc=tokenlearner_loc,
                                       fuse=fuse_module, dropout=dropout)

        self.num_classes = num_classes
        self.fusion = fusion
        self.pool = pool
        self.to_latent = nn.Identity()
        self.ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, img, single_modality=False, labels=None):
        # img = img['RGB']
        if self.fusion is True:
            x = img
            x = x.transpose(1, 2)
        else:
            b, c, w, h = img.shape
            img = img.reshape(b * 8, c // 8, w, h)
            x = self.embedding(img)  # b, hidden_size, w, h
            b, c, w, h = x.shape
            x = x.reshape(b // 8, c, 8 * w * h)
            x += self.pos_embedding
            x = self.dropout(x)

        x = self.transformer(x)
        if single_modality is True:
            return x
        else:
            # mean
            x = x.mean(dim=1)

            x = self.to_latent(x)
            x = self.ln(x)

            logits = self.head(x)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
                return loss
            else:
                return logits


class complete_model(nn.Module):
    def __init__(self, num_classes=10, zero_head=True):
        super(complete_model, self).__init__()
        self.model_visual = ViT(image_size=224, num_tokens=8, fuse=False, v11=True, tokenlearner_loc=8, patch_size=16,
                                num_classes=num_classes, in_channels=3, hidden_size=768, depth=12, heads=16, mlp_dim=2048,
                                dropout=0.1, emb_dropout=0.1)
        self.model_audio = ViT(image_size=128, num_tokens=8, fuse=False, v11=True, tokenlearner_loc=5, patch_size=16,
                               num_classes=num_classes, in_channels=1, hidden_size=768, depth=8, heads=16, mlp_dim=2048,
                               dropout=0.1, emb_dropout=0.1)
        self.model_fusion = ViT(image_size=128, num_tokens=8, fuse=False, v11=True, tokenlearner_loc=4, patch_size=16,
                                num_classes=num_classes, in_channels=3, hidden_size=768, depth=4, heads=16, mlp_dim=2048, dropout=0.1,
                                emb_dropout=0.1, fusion=True)
        self.zero_head = zero_head

    def forward(self, x, labels=None):
        tokens_visual = self.model_visual(x['RGB'], labels=labels, single_modality=False)
        # tokens_audio = self.model_audio(x['Spec'], single_modality=True)
        # tokens_fusion = torch.cat((tokens_visual, tokens_audio), dim=1)
        # output = self.model_fusion(tokens_fusion, labels=labels)
        output = tokens_visual
        return output

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.model_visual.head.weight)
                nn.init.zeros_(self.model_visual.head.bias)
            else:
                self.model_visual.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.model_visual.head.bias.copy_(np2th(weights["head/bias"]).t())
            weights_audio = np.sum(weights["embedding/kernel"], axis=2)
            weights_audio = np.expand_dims(weights_audio, 2)
            self.model_audio.embedding.weight.copy_(np2th(weights_audio, conv=True))
            self.model_audio.embedding.bias.copy_(np2th(weights["embedding/bias"]))
            self.model_visual.embedding.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.model_visual.embedding.bias.copy_(np2th(weights["embedding/bias"]))
