#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
import collections
import math
import random
from kaiwu_agent.utils.common_func import attached, create_cls
from agent_diy.conf.conf import Config

# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)

RelativeDistance = {
    "RELATIVE_DISTANCE_NONE": 0,
    "VerySmall": 1,
    "Small": 2,
    "Medium": 3,
    "Large": 4,
    "VeryLarge": 5,
}


RelativeDirection = {
    "East": 1,
    "NorthEast": 2,
    "North": 3,
    "NorthWest": 4,
    "West": 5,
    "SouthWest": 6,
    "South": 7,
    "SouthEast": 8,
}

DirectionAngles = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180,
    6: 225,
    7: 270,
    8: 315,
}

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, sample_data):
        self.buffer.append(sample_data)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return transitions

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

def calculate_distance(pos1, pos2):
    if pos1 is None or pos2 is None:
        return None
    return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

def reward_process(raw_reward, agent_pos, prev_pos, nearest_treasure_pos, end_pos,
                prev_dist_to_treasure, prev_dist_to_end, current_steps, is_terminal,
                is_bad_action=False, is_flash_used=False, is_truncated):
    """
    Args:
        raw_reward (float): 环境返回的原始奖励。
        agent_pos (tuple): 智能体当前 (x, z) 坐标。
        prev_pos (tuple): 智能体上一帧 (x, z) 坐标。
        nearest_treasure_pos (tuple/None): 最近宝箱的 (x, z) 坐标或 None。
        end_pos (tuple/None): 终点的 (x, z) 坐标或 None。
        prev_dist_to_treasure (float/None): 上一帧智能体到最近宝箱的距离。
        prev_dist_to_end (float/None): 上一帧智能体到终点的距离。
        current_steps (int): 当前游戏步数。
        is_terminal (bool): 当前步是否是回合终止。
        is_bad_action (bool): 当前动作是否被识别为无效动作（如原地踏步）。
        is_flash_used (bool): 闪现是否可用。
        is_truncated: 游戏是否失败
    Returns:
        list [float]: 经过塑形处理后的最终奖励。
    """
    processed_reward = 0.0
    processed_reward += raw_reward * Config.REWARD_SCALE_TERMINAL

    # 1. 最终奖励处理 (如果回合结束，对原始奖励进行缩放)
    if is_terminal:
        if is_truncated:
            processed_reward -= Config.REWARD_FAILED_PENALTY
        return [processed_reward]

    # 2. 时间惩罚：每一步一个小额负奖励
    processed_reward -= Config.REWARD_TIME_PENALTY

    # 3. 无效行动惩罚
    if is_bad_action:
        processed_reward -= Config.REWARD_BAD_ACTION_PENALTY
        
    # 4. 闪现奖励：只有当智能体使用了闪现时才计算
    if is_flash_used:
        # 确定目标，计算闪现带来的距离变化
        target_pos = nearest_treasure_pos if nearest_treasure_pos is not None else end_pos
        if target_pos:
            dist_before_flash = calculate_distance(prev_pos, target_pos)
            dist_after_flash = calculate_distance(agent_pos, target_pos)
            flash_distance_change = dist_before_flash - dist_after_flash
            
            if flash_distance_change > 0:
                processed_reward += flash_distance_change * Config.REWARD_SCALE_FLASH_DIST
            else:
                processed_reward -= Config.REWARD_PENALTY_BAD_FLASH

    # 5. 普通移动奖励：当没有使用闪现时才计算
    else: # is_flash_used == False
        # 距离奖励 (优先级：宝箱 > 终点)
        if nearest_treasure_pos is not None:
            current_dist_to_treasure = calculate_distance(agent_pos, nearest_treasure_pos)
            if prev_dist_to_treasure is not None:
                distance_change_treasure = prev_dist_to_treasure - current_dist_to_treasure
                processed_reward += distance_change_treasure * Config.REWARD_SCALE_TREASURE_DIST
        
        elif end_pos is not None:
            current_dist_to_end = calculate_distance(agent_pos, end_pos)
            if prev_dist_to_end is not None:
                distance_change_end = prev_dist_to_end - current_dist_to_end
                processed_reward += distance_change_end * Config.REWARD_SCALE_END_DIST
    return [processed_reward]

@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


@attached
def SampleData2NumpyData(g_data):
    # 确保所有数据都是一维数组，方便 hstack 拼接
    obs_arr = np.array(g_data.obs, dtype=np.float32).flatten()
    _obs_arr = np.array(g_data._obs, dtype=np.float32).flatten()
    obs_legal_arr = np.array(g_data.obs_legal, dtype=np.float32).flatten() # 16维
    _obs_legal_arr = np.array(g_data._obs_legal, dtype=np.float32).flatten() # 16维
    act_arr = np.array([g_data.act], dtype=np.float32) # 确保是 [act] 变成 (1,)
    rew_arr = np.array([g_data.rew], dtype=np.float32) # 变成 (1,)
    ret_arr = np.array([g_data.ret], dtype=np.float32) # 变成 (1,)
    done_arr = np.array([g_data.done], dtype=np.float32) # 变成 (1,)

    return np.hstack(
        (
            obs_arr,
            _obs_arr,
            obs_legal_arr,
            _obs_legal_arr,
            act_arr,
            rew_arr,
            ret_arr,
            done_arr,
        )
    )

@attached
def NumpyData2SampleData(s_data):
    obs_data_size = Config.DIM_OF_OBSERVATION
    # 之前的问题在于这里使用了 DIM_OF_ACTION_DIRECTION (8)
    # legal_data_size = Config.DIM_OF_ACTION_DIRECTION

    # 应该使用总的动作空间维度 (16)
    total_action_space_size = Config.TOTAL_ACTION_SPACE # 确保这个值为 16

    # 定义每个部分的起始和结束索引，使其与 SampleData2NumpyData 的拼接顺序和大小一致

    # obs 和 _obs 部分的长度
    current_idx = 0
    obs_end_idx = current_idx + obs_data_size
    obs = s_data[current_idx : obs_end_idx]
    current_idx = obs_end_idx

    _obs_end_idx = current_idx + obs_data_size
    _obs = s_data[current_idx : _obs_end_idx]
    current_idx = _obs_end_idx

    # obs_legal 部分的长度 - 应该使用 total_action_space_size (16)
    obs_legal_end_idx = current_idx + total_action_space_size
    obs_legal = s_data[current_idx : obs_legal_end_idx]
    current_idx = obs_legal_end_idx

    # _obs_legal 部分的长度 - 应该使用 total_action_space_size (16)
    _obs_legal_end_idx = current_idx + total_action_space_size
    _obs_legal = s_data[current_idx : _obs_legal_end_idx]
    current_idx = _obs_legal_end_idx

    act = s_data[current_idx]
    rew = s_data[current_idx + 1]
    ret = s_data[current_idx + 2]
    done = s_data[current_idx + 3]

    act = s_data[-4]
    rew = s_data[-3]
    ret = s_data[-2]
    done = s_data[-1]


    return SampleData(
        obs=obs,
        _obs=_obs,
        obs_legal=obs_legal,
        _obs_legal=_obs_legal,
        act=act,
        rew=rew,
        ret=ret,
        done=done,
    )