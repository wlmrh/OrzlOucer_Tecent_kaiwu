#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_target_dqn.conf.conf import Config

# Adjust thread settings based on entry point
if os.path.basename(sys.argv[0]) == "learner.py":
    torch.set_num_interop_threads(2)
    torch.set_num_threads(2)
else:
    torch.set_num_interop_threads(4)
    torch.set_num_threads(4)


def make_fc_layer(in_features: int, out_features: int) -> nn.Linear:
    """
    Create and initialize a linear layer with orthogonal weights and zero bias.
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight)
    nn.init.zeros_(fc.bias)
    return fc


class ResidualBlock(nn.Module):
    """
    A simple residual block for MLP with two linear layers.
    """
    def __init__(self, dim: int):
        super(ResidualBlock, self).__init__()
        self.fc1 = make_fc_layer(dim, dim)
        self.fc2 = make_fc_layer(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.relu(out + identity)


class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        # feature configure parameter
        # 特征配置参数
        self.feature_len = Config.DIM_OF_OBSERVATION

        # Q network
        # Q 网络
        self.q_mlp = MLP([self.feature_len, 256, 128, action_shape], "q_mlp")

    # Forward inference
    # 前向推理
    def forward(self, feature):
        # Action and value processing
        logits = self.q_mlp(feature)
        return logits