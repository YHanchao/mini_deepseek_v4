"""
Implement model blocks for deepseek

DeepSeek似乎很多地方都要做混合精度，需要注意不能直接全都bf16或者fp32进去
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSMoE(nn.Module):
    def __init__(self):
        pass
