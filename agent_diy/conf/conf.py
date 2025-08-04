#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration, including dimension settings, algorithm parameter settings.
# The last few configurations in the file are for the Kaiwu platform to use and should not be changed.
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:
    ACTION_LEN = 1 # 保持不变，代表一个动作的长度（例如，一个整数索引）
    ACTION_NUM = 16 # 保持不变，总动作数量 8 移动 + 8 闪现

    # --- 局部网格观测配置 ---
    LOCAL_GRID_HEIGHT = 11
    LOCAL_GRID_WIDTH = 11
    LOCAL_GRID_CHANNELS = 5
    # 局部网格展平后的维度 (11 * 11 * 5 = 605)
    LOCAL_GRID_FLAT_DIM = LOCAL_GRID_HEIGHT * LOCAL_GRID_WIDTH * LOCAL_GRID_CHANNELS

    # --- 辅助特征维度 ---
    NUM_AUX_FEATURES = 16

    # features: 现在 FEATURES 列表用来表示新观测中各个部分的维度
    # 注意: 这里我们将 FEATURES 定义为包含两个主要部分的维度
    # 第一个元素是局部网格展平后的维度 (605)
    # 第二个元素是所有辅助特征的总维度 (15)
    # 这样 sum(FEATURES) 就会得到我们需要的总观测维度
    FEATURES = [
        LOCAL_GRID_FLAT_DIM, # 局部视野网格展平后的维度 (11 * 11 * 5)
        1,  # 闪现是否可用
        8,  # onehot 每个宝箱是否已被获得
        1,  # 归一化当前的步数
        2,  # 归一化后的当前位置坐标
        2,  # 归一化后的最近宝箱（大致）坐标
        2,   # 归一化后的终点（大致）坐标
    ]

    FEATURE_SPLIT_SHAPE = FEATURES # FEATURE_SPLIT_SHAPE 保持与 FEATURES 一致

    # Size of observation
    # observation的维度
    # 此时 DIM_OF_OBSERVATION = 605 + 16 = 621
    DIM_OF_OBSERVATION = sum(FEATURES)

    # --- 动作空间配置 ---
    # 移动动作方向的维度
    DIM_OF_ACTION_DIRECTION = 8

    # 闪现动作方向的维度
    DIM_OF_TALENT = 8
    
    # 总动作空间维度 (8个移动 + 8个闪现 = 16)
    TOTAL_ACTION_SPACE = DIM_OF_ACTION_DIRECTION + DIM_OF_TALENT

    # 最大步数
    MAX_STEP_NO = 2000

    # Input dimension of reverb sample on learner. Note that different algorithms have different dimensions.
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    # 这里我们重新计算 SAMPLE_DIM 以反映新的观测维度和合法动作掩码
    # FRAME 结构: obs (620), _obs (620), act (1), rew (1), done (1), ret (1), obs_legal (16), _obs_legal (16)
    # 总和: 614 * 2 + 1 + 1 + 1 + 1 + 16 * 2 = 1228 + 4 + 32 = 1264
    SAMPLE_DIM = (DIM_OF_OBSERVATION * 2) + 4 + (TOTAL_ACTION_SPACE * 2)


    # Update frequency of target network
    # target网络的更新频率
    TARGET_UPDATE_FREQ = 200

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.9

    # epsilon
    EPSILON_MIN = 0.1
    EPSILON_MAX = 1.0
    EPSILON_DECAY = 1e-7

    # Initial learning rate
    # 初始的学习率
    START_LR = 1e-4
    LR_DECAY_RATE = 0.00001

    # --- 奖励函数相关系数 ---
    # 1. 最终奖励与失败惩罚
    # 将环境的原始奖励（如100分）缩放到与每一步的奖励在同一量级
    REWARD_SCALE_TERMINAL = 0.01

    # 2. 核心奖励信号
    # 每收集一个宝箱，给予的额外奖励。这个奖励是持续的，每一步都会得到
    REWARD_TREASURE_BONUS = 0.2
    # 靠近宝箱的距离奖励，应该略高于终点奖励，以鼓励前期探索
    REWARD_SCALE_TREASURE_DIST = 0.05
    # 靠近终点的距离奖励，会乘上一个小于1的权重
    REWARD_SCALE_END_DIST = 0.1

    # 3. 动作奖励与惩罚
    # 闪现奖励，比普通移动略低，避免刷分
    REWARD_SCALE_FLASH_DIST = 0.03
    # 乱用闪现的惩罚要大，让模型学会谨慎使用
    REWARD_PENALTY_BAD_FLASH = 0.5
    # 每一步的时间惩罚，用于鼓励智能体快速行动，不要在原地徘徊
    REWARD_TIME_PENALTY = 0.05
    # 低效行动的惩罚，鼓励智能体采取更有效的行动
    REWARD_BAD_ACTION_PENALTY = 0.05
    # 访问次数惩罚，鼓励智能体探索新区域
    REPEAT_VISIT_PENALTY = 0.05
    
    # 内存池相关参数
    REPLAY_BUFFER_SIZE = 30000     # 内存池最大容量
    REPLAY_BUFFER_MIN_SIZE = 300   # 内存池开始学习的最小容量 (例如 BATCH_SIZE * 10)
    BATCH_SIZE = 32                 # 每次从内存池采样的数据量
    TRAIN_ITERATIONS_PER_EPISODE = 5 # 每个回合结束后，从内存池学习的次数