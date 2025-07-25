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
    """
    DQN-based Q-network with optional Dueling architecture and Residual MLP.

    Args:
        state_dim (int): Length of the input feature vector.
        action_shape (int): Number of discrete actions.
        dueling (bool): Whether to use Dueling DQN.
        num_res_blocks (int): Number of residual blocks in MLP.
        device (Optional[torch.device]): Device for computation.
    """
    def __init__(
        self,
        state_dim: int,
        action_shape: int,
        dueling: bool = False,
        num_res_blocks: int = 2,
        device: Optional[torch.device] = None,
    ):
        super(Model, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dueling = dueling
        self.state_dim = state_dim
        self.action_shape = action_shape
        self.num_res_blocks = num_res_blocks

        # Residual MLP feature encoder
        hidden_dim = 256
        self.fc_in = make_fc_layer(state_dim, hidden_dim)
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])

        if self.dueling:
            # shared representation after residual blocks
            self.fc_shared = make_fc_layer(hidden_dim, hidden_dim)
            # Value stream
            self.value_stream = nn.Sequential(
                make_fc_layer(hidden_dim, hidden_dim),
                nn.ReLU(),
                make_fc_layer(hidden_dim, 1)
            )
            # Advantage stream
            self.adv_stream = nn.Sequential(
                make_fc_layer(hidden_dim, hidden_dim),
                nn.ReLU(),
                make_fc_layer(hidden_dim, action_shape)
            )
        else:
            # final output layer
            self.fc_out = make_fc_layer(hidden_dim, action_shape)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for a batch of state feature vectors.

        Args:
            x (torch.Tensor): shape [B, state_dim]
        Returns:
            q (torch.Tensor): shape [B, action_shape]
        """
        x = x.to(self.device).float()
        out = F.relu(self.fc_in(x))
        out = self.res_blocks(out)  # apply residual blocks

        if self.dueling:
            shared = F.relu(self.fc_shared(out))
            val = self.value_stream(shared)            # [B,1]
            adv = self.adv_stream(shared)             # [B, A]
            adv_mean = adv.mean(dim=1, keepdim=True)
            q = val + adv - adv_mean
        else:
            q = self.fc_out(out)
        return q

# Example instantiation:
# model = Model(
#     state_dim=Config.DIM_OF_OBSERVATION,
#     action_shape=8,
#     dueling=True,
#     num_res_blocks=3
# ).to(model.device)
