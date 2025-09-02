# Path: pyramid_vig.py
# This version includes the "Attention-Augmented" classifier head (Strategy 1).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model
from gcn_lib import Grapher, act_layer

# --- HELPER FUNCTIONS AND CONFIGS (UNCHANGED) ---
def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head', **kwargs
    }

default_cfgs = {
    'vig_224_gelu': _cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vig_b_224_gelu': _cfg(crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}


# <--- START: NEW BLOCK FOR ATTENTION-AUGMENTED CLASSIFIER ---

# SE Block (Adapted to use 1x1 Convs, which is equivalent to Linear for this purpose)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale

# <--- END: NEW BLOCK ---


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0), nn.BatchNorm2d(hidden_features))
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0), nn.BatchNorm2d(out_features))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class Stem(nn.Module):
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim//2), act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim), act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1), nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.convs(x)

class Downsample(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim))

    def forward(self, x):
        return self.conv(x)

class DeepGCN(torch.nn.Module):
    # <--- CHANGE: Added 'classifier_type' argument for toggling ---
    def __init__(self, opt, classifier_type='original'):
        super(DeepGCN, self).__init__()
        k, act, norm, bias = opt.k, opt.act, opt.norm, opt.bias
        epsilon, stochastic, conv, drop_path, blocks, channels = opt.epsilon, opt.use_stochastic, opt.conv, opt.drop_path, opt.blocks, opt.channels
        self.n_blocks = sum(blocks)
        reduce_ratios, dpr, num_knn = [4, 2, 1, 1], [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)], [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]
        max_dilation = 49 // max(num_knn)
        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224//4, 224//4))
        HW = 224 // 4 * 224 // 4
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW //= 4
            for j in range(blocks[i]):
                self.backbone.append(Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm, bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx], relative_pos=True), FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])))
                idx += 1
        self.backbone = Seq(*self.backbone)

        # <--- CHANGE: Classifier Selection Logic ---
        if classifier_type == 'attention_augmented':
            print("Using Attention-Augmented Classifier Head.")
            self.prediction = Seq(
                nn.Conv2d(channels[-1], 1024, 1, bias=True),
                nn.BatchNorm2d(1024),
                act_layer(act),
                SEBlock(1024),  # The new attention block
                nn.Dropout(opt.dropout),
                nn.Conv2d(1024, opt.n_classes, 1, bias=True)
            )
        else: # Default to the original
            print("Using Original Classifier Head.")
            self.prediction = Seq(
                nn.Conv2d(channels[-1], 1024, 1, bias=True),
                nn.BatchNorm2d(1024),
                act_layer(act),
                nn.Dropout(opt.dropout),
                nn.Conv2d(1024, opt.n_classes, 1, bias=True)
            )
        
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        # The forward pass is unchanged for this strategy
        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

@register_model
def pvig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k, self.conv, self.act, self.norm, self.bias, self.dropout, self.use_dilation, self.epsilon, self.use_stochastic, self.drop_path = 9, 'mr', 'gelu', 'batch', True, 0.0, True, 0.2, False, drop_path_rate
            self.blocks, self.channels, self.n_classes, self.emb_dims = [2,2,6,2], [48, 96, 240, 384], num_classes, 1024
    opt = OptInit(**kwargs)
    model = DeepGCN(opt, classifier_type='attention_augmented') # Uses new head by default
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model

# --- NEW CUSTOMIZABLE MODEL ---
@register_model
def pvig_ti_custom(pretrained=False, **kwargs):
    """
    A custom variant of pvig_ti that you can easily modify for experiments.
    """
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.conv, self.act, self.norm, self.bias, self.dropout, self.use_dilation, self.epsilon, self.use_stochastic, self.drop_path = 'mr', 'gelu', 'batch', True, 0.0, True, 0.2, False, drop_path_rate
            self.n_classes = num_classes
            self.emb_dims = 1024
            self.k = 9
            self.blocks = [2, 2, 6, 2]
            self.channels = [48, 96, 240, 384]

    opt = OptInit(**kwargs)
    # <--- CHANGE: Select the classifier when the model is created ---
    # Set this to 'original' to use the old head for comparison experiments.
    model = DeepGCN(opt, classifier_type='attention_augmented')
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model
# -----------------------------

@register_model
def pvig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k, self.conv, self.act, self.norm, self.bias, self.dropout, self.use_dilation, self.epsilon, self.use_stochastic, self.drop_path = 9, 'mr', 'gelu', 'batch', True, 0.0, True, 0.2, False, drop_path_rate
            self.blocks, self.channels, self.n_classes, self.emb_dims = [2,2,6,2], [80, 160, 400, 640], num_classes, 1024
    opt = OptInit(**kwargs)
    model = DeepGCN(opt, classifier_type='attention_augmented')
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model

@register_model
def pvig_m_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k, self.conv, self.act, self.norm, self.bias, self.dropout, self.use_dilation, self.epsilon, self.use_stochastic, self.drop_path = 9, 'mr', 'gelu', 'batch', True, 0.0, True, 0.2, False, drop_path_rate
            self.blocks, self.channels, self.n_classes, self.emb_dims = [2,2,16,2], [96, 192, 384, 768], num_classes, 1024
    opt = OptInit(**kwargs)
    model = DeepGCN(opt, classifier_type='attention_augmented')
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model

@register_model
def pvig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k, self.conv, self.act, self.norm, self.bias, self.dropout, self.use_dilation, self.epsilon, self.use_stochastic, self.drop_path = 9, 'mr', 'gelu', 'batch', True, 0.0, True, 0.2, False, drop_path_rate
            self.blocks, self.channels, self.n_classes, self.emb_dims = [2,2,18,2], [128, 256, 512, 1024], num_classes, 1024
    opt = OptInit(**kwargs)
    model = DeepGCN(opt, classifier_type='attention_augmented')
    model.default_cfg = default_cfgs['vig_b_224_gelu']
    return model