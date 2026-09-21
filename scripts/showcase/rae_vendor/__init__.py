"""Vendored from https://github.com/bytetriper/RAE (MIT, Boyang Zheng 2025): src/stage1/decoders/.
Only the ViT decoder; the encoder is taken from torch hub (dinov2_vitb14_reg) because the
cluster transformers build predates Dinov2WithRegistersModel."""
from .decoder import GeneralDecoder
from .utils import ViTMAEConfig
