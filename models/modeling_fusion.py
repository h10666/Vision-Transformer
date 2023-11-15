# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3, num_frames=8):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.num_frames = num_frames
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16) * num_frames
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * num_frames
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = x.view((B * self.num_frames, x.shape[1] // self.num_frames) + x.shape[-2:])
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = x.reshape(B, x.shape[1] * self.num_frames, x.shape[-1])
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, num_layers, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, in_channels, fusion, num_layers, vis, num_frames):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels, num_frames=num_frames)
        self.encoder = Encoder(config, num_layers, vis)
        self.fusion = fusion

    def forward(self, input_ids):
        if self.fusion is False:
            embedding_output = self.embeddings(input_ids)
        else:
            embedding_output = input_ids
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, in_channels=3, num_layers=12, fusion=False, num_classes=21843,
                 zero_head=False, vis=False, num_frames=8):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, in_channels, fusion, num_layers, vis, num_frames)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        if labels is None:
            return x
        else:
            embedding = x[:, 0].view(x[:, 0].shape[0] // 8, 8, self.config.hidden_size)       # B*8, 768 -> B, 8, 768
            embedding = torch.mean(embedding, dim=1)        # B, 8, 768 -> B, 768
            logits = self.head(embedding)                   # B, 768 -> B, 10
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return logits

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())
            weights_audio = np.sum(weights["embedding/kernel"], axis=2)
            weights_audio = np.expand_dims(weights_audio, 2)
            self.transformer.embeddings_audio.patch_embeddings.weight.copy_(np2th(weights_audio, conv=True))
            self.transformer.embeddings_audio.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings_audio.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.embeddings_visual.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings_visual.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings_visual.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_visual_new = self.transformer.embeddings_visual.position_embeddings
            posemb_audio_new = self.transformer.embeddings_audio.position_embeddings

            if posemb.size() == posemb_visual_new.size():
                self.transformer.embeddings_visual.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_visual_new.size()))
                ntok_new = posemb_visual_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[:, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[:]

                num_frames = ntok_new // posemb_grid.size(1)
                new_posemb_grid = posemb_grid
                for i in range(num_frames - 1):
                    new_posemb_grid = torch.cat((new_posemb_grid, posemb_grid), 1)

                posemb = np.concatenate([posemb_tok, new_posemb_grid], axis=1)
                self.transformer.embeddings_visual.position_embeddings.copy_(np2th(posemb))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            if posemb.size() == posemb_audio_new.size():
                self.transformer.embeddings_audio.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_audio_new.size()))
                ntok_new = posemb_audio_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new // num_frames))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                new_posemb_grid = posemb_grid
                for i in range(num_frames - 1):
                    new_posemb_grid = np.concatenate([new_posemb_grid, posemb_grid], axis=1)
                posemb = np.concatenate([posemb_tok, new_posemb_grid], axis=1)
                self.transformer.embeddings_audio.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings_visual.hybrid:
                self.transformer.embeddings_visual.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings_visual.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings_visual.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings_visual.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

            if self.transformer.embeddings_audio.hybrid:
                self.transformer.embeddings_audio.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings_audio.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings_audio.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings_audio.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}


class complete_model(nn.Module):
    def __init__(self, config, num_classes=21843, zero_head=True):
        super(complete_model, self).__init__()
        self.model_visual = VisionTransformer(config, img_size=224, in_channels=3, num_layers=12, fusion=False, num_classes=num_classes,
                                              zero_head=True, num_frames=1)
        self.model_audio = VisionTransformer(config, img_size=128, in_channels=1, num_layers=12, fusion=False, num_classes=num_classes,
                                             zero_head=True, num_frames=1)
        self.model_fusion = VisionTransformer(config, num_layers=4, fusion=True, zero_head=True,
                                              num_classes=num_classes)
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.head = Linear(config.hidden_size, num_classes)
        self.num_classes = num_classes

    def forward(self, x, labels=None, is_train=False):
        """
        B = x['RGB'].shape[0]
        x['RGB'] = x['RGB'].view((B * 8, x['RGB'].shape[1] // 8) + x['RGB'].shape[-2:])
        x['Spec'] = x['Spec'].view((B * 8, x['Spec'].shape[1] // 8) + x['Spec'].shape[-2:])

        tokens_visual = self.model_visual(x['RGB']) # B*8, 3, w, h -> B*8, 196+1, 768
        tokens_audio = self.model_audio(x['Spec'])  # B*8, 3, w, h -> B*8, 64+1, 768
        tokens_classifier = torch.add(tokens_visual[:, 0], tokens_audio[:, 0])
        tokens_classifier = tokens_classifier.unsqueeze(1)  # B*8, 1, 768
        tokens_fusion = torch.cat((tokens_visual[:, 1:], tokens_audio[:, 1:]), dim=1)  # B*8, 260, 768
        tokens_fusion = torch.cat((tokens_classifier, tokens_fusion), dim=1)   # B*8, 261, 768
        output = self.model_fusion(tokens_fusion, labels)   # (B, 10), loss

        return output
        """
        B = x['RGB'].shape[0]
        x['RGB'] = x['RGB'].view((B * 8, x['RGB'].shape[1] // 8) + x['RGB'].shape[-2:])
        x['Spec'] = x['Spec'].view((B * 8, x['Spec'].shape[1] // 8) + x['Spec'].shape[-2:])
        logits_visual = self.model_visual(x['RGB'], labels=labels)
        logits_audio = self.model_audio(x['Spec'], labels=labels)
        logits = (logits_visual + logits_audio) / 2

        if is_train is True:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.model_fusion.head.weight)
                nn.init.zeros_(self.model_fusion.head.bias)
                nn.init.zeros_(self.model_visual.head.weight)
                nn.init.zeros_(self.model_visual.head.bias)
                nn.init.zeros_(self.model_audio.head.weight)
                nn.init.zeros_(self.model_audio.head.bias)
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.model_fusion.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.model_fusion.head.bias.copy_(np2th(weights["head/bias"]).t())
            weights_audio = np.sum(weights["embedding/kernel"], axis=2)
            weights_audio = np.expand_dims(weights_audio, 2)
            self.model_audio.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights_audio, conv=True))
            self.model_audio.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.model_audio.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.model_visual.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True))
            self.model_visual.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.model_visual.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))

            self.model_visual.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"]))
            self.model_visual.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"]))
            self.model_audio.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"]))
            self.model_audio.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"]))
            self.model_fusion.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"]))
            self.model_fusion.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_visual_new = self.model_visual.transformer.embeddings.position_embeddings
            posemb_audio_new = self.model_audio.transformer.embeddings.position_embeddings

            if posemb.size() == posemb_visual_new.size():
                self.model_visual.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_visual_new.size()))
                ntok_new = posemb_visual_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[:, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[:]

                num_frames = ntok_new // posemb_grid.size(1)
                new_posemb_grid = posemb_grid
                for i in range(num_frames - 1):
                    new_posemb_grid = torch.cat((new_posemb_grid, posemb_grid), 1)

                posemb = np.concatenate([posemb_tok, new_posemb_grid], axis=1)
                self.model_visual.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            if posemb.size() == posemb_audio_new.size():
                self.model_audio.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_audio_new.size()))
                ntok_new = posemb_audio_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                new_posemb_grid = posemb_grid
                posemb = np.concatenate([posemb_tok, new_posemb_grid], axis=1)
                self.model_audio.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.model_visual.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            for bname, block in self.model_audio.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
