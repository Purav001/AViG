# Path: hubconf.py
import warnings
import torch

dependencies = ['torch', 'timm']

# --- Model Imports ---
# These models are defined in the local project files.
import vig
import pyramid_vig

# NOTE: SNNMLP and GhostNet have been removed to create a self-contained project
# that does not require cloning external repositories.

# --- ViG Model Entrypoints ---

def vig_ti_224_gelu(pretrained: bool = False, **kwargs):
    if pretrained:
        warnings.warn("Pretrained weights are not available for this model via torch.hub.")
    model = vig.vig_ti_224_gelu(pretrained=False, **kwargs)
    return model

def vig_s_224_gelu(pretrained: bool = False, **kwargs):
    if pretrained:
        warnings.warn("Pretrained weights are not available for this model via torch.hub.")
    model = vig.vig_s_224_gelu(pretrained=False, **kwargs)
    return model

def vig_b_224_gelu(pretrained: bool = False, **kwargs):
    if pretrained:
        warnings.warn("Pretrained weights are not available for this model via torch.hub.")
    model = vig.vig_b_224_gelu(pretrained=False, **kwargs)
    return model

# --- Pyramid-ViG Model Entrypoints ---

def pvig_ti_224_gelu(pretrained: bool = False, **kwargs):
    if pretrained:
        warnings.warn("Pretrained weights are not available for this model via torch.hub.")
    model = pyramid_vig.pvig_ti_224_gelu(pretrained=False, **kwargs)
    return model

def pvig_s_224_gelu(pretrained: bool = False, **kwargs):
    if pretrained:
        warnings.warn("Pretrained weights are not available for this model via torch.hub.")
    model = pyramid_vig.pvig_s_224_gelu(pretrained=False, **kwargs)
    return model

def pvig_m_224_gelu(pretrained: bool = False, **kwargs):
    if pretrained:
        warnings.warn("Pretrained weights are not available for this model via torch.hub.")
    model = pyramid_vig.pvig_m_224_gelu(pretrained=False, **kwargs)
    return model

def pvig_b_224_gelu(pretrained: bool = False, **kwargs):
    if pretrained:
        warnings.warn("Pretrained weights are not available for this model via torch.hub.")
    model = pyramid_vig.pvig_b_224_gelu(pretrained=False, **kwargs)
    return model