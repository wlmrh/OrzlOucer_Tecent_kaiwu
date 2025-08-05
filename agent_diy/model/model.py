#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import List
from agent_diy.conf.conf import Config

import sys
import os

if os.path.basename(sys.argv[0]) == "learner.py":
    import torch

    torch.set_num_interop_threads(2)
    torch.set_num_threads(2)
else:
    import torch

    torch.set_num_interop_threads(4)
    torch.set_num_threads(4)


# --- 辅助函数：创建并初始化线性层 ---
def make_fc_layer(in_features: int, out_features: int):
    # Wrapper function to create and initialize a linear layer
    # 创建并初始化一个线性层
    fc_layer = nn.Linear(in_features, out_features)

    # initialize weight and bias
    # 初始化权重及偏移量
    nn.init.orthogonal_(fc_layer.weight) # 使用_表示原地操作
    nn.init.zeros_(fc_layer.bias)

    return fc_layer

# --- 辅助函数：创建 MLP ---
class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        # Create a MLP object
        # 创建一个 MLP 对象
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # no relu for the last fc layer of the mlp unless required
            # 除非有需要，否则 mlp 的最后一个 fc 层不使用 relu
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)


# --- 主要模型类 ---
class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        # feature configure parameter
        # 特征配置参数
        # state_shape 是你的 (11, 11, 5) 或者其他包含所有信息元组
        # 这里假设 state_shape 是一个元组 (H, W, C) 或者直接是展平后的总维度
        # 为了兼容性，我们要求传入的 feature 已经是扁平化的
        self.total_input_dim = state_shape # feature 的总维度

        # 定义局部视野网格的维度
        # 假设局部网格是 11x11x5，其展平后是 11*11*5 = 605
        self.local_grid_height = Config.LOCAL_GRID_HEIGHT
        self.local_grid_width = Config.LOCAL_GRID_WIDTH
        self.local_grid_channels = Config.LOCAL_GRID_CHANNELS
        self.cnn_input_dim = self.local_grid_height * self.local_grid_width * self.local_grid_channels
        self.cnn_output_dim = 64 * 7 * 7
        # 定义辅助特征的维度
        self.n_aux_features = Config.NUM_AUX_FEATURES

        # 检查传入的 state_shape 是否与预期一致
        if self.total_input_dim != (self.cnn_input_dim + self.n_aux_features):
            raise ValueError(f"Expected state_shape (total_input_dim) to be {self.cnn_input_dim + self.n_aux_features} (CNN_flat + Aux_features), but got {self.total_input_dim}")

        # --- 局部视野处理流 (CNN Stream) ---
        # CNN 输入: (batch_size, channels, height, width)
        self.conv1 = nn.Conv2d(in_channels=self.local_grid_channels, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        # --- 辅助特征处理流 (Auxiliary FC Stream) ---
        self.aux_mlp = MLP(
            [self.n_aux_features, 64], # 输入16维，映射到64维
            "aux_mlp",
            non_linearity_last=False # 辅助流的最后一层也可以有激活函数，具体看效果
        )

        # --- 特征融合与决策流 (Combined FC Stream) ---
        self.combined_input_dim = self.cnn_output_dim + 64 # 3136 (来自CNN) + 64 (来自辅助MLP)
        self.q_mlp = MLP(
            [self.combined_input_dim, 128, 64, action_shape],
            "q_mlp"
        )

    # Forward inference
    # 前向推理
    def forward(self, feature):
        # 假设 feature 是已经拼接好的扁平化张量
        # 首先将其拆分成局部网格特征和辅助特征

        # 1. 提取局部网格特征并重塑为 CNN 输入格式
        # feature 的前 self.cnn_input_dim 维度是局部网格展平后的数据
        local_grid_flat = feature[:, :self.cnn_input_dim]
        # 重塑为 (batch_size, channels, height, width)
        # 注意: 如果你的原始数据是 (H, W, C)，展平后需要重新排列
        # PyTorch 的 Conv2d 要求输入是 (N, C, H, W)
        local_grid_obs = local_grid_flat.view(
            -1, self.local_grid_channels, self.local_grid_height, self.local_grid_width
        )

        # 2. 提取辅助特征
        # feature 的剩余维度是辅助特征
        aux_features = feature[:, self.cnn_input_dim:]

        # --- 局部视野处理流 (CNN Stream) ---
        x_cnn = F.relu(self.conv1(local_grid_obs))
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = x_cnn.view(-1, self.cnn_output_dim) # 展平

        # --- 辅助特征处理流 (Auxiliary FC Stream) ---
        x_aux = self.aux_mlp(aux_features) # MLP 内部已包含激活函数

        # --- 特征融合与决策流 (Combined FC Stream) ---
        x_combined = torch.cat((x_cnn, x_aux), dim=1) # 沿着特征维度拼接

        # Action and value processing
        logits = self.q_mlp(x_combined) # MLP 内部已包含激活函数，最后一层除外

        return logits