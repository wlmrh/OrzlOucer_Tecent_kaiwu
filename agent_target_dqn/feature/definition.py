#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.utils.common_func import attached, create_cls
from agent_target_dqn.conf.conf import Config

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


def reward_process(end_dist, history_dist, map_info=None, is_treasure_collected=False, is_goal_reached=False, 
                  remaining_steps=None, max_steps=None, last_distance_to_goal=None, current_distance_to_goal=None):
    """
    新的奖励函数实现
    包含时间惩罚、距离奖励、墙壁惩罚和动态价值宝箱
    """
    # 1. 初始化当前步的奖励
    step_reward = 0

    # 2. 时间惩罚 (鼓励效率)
    step_reward -= 0.2

    # 3. 终点距离变化奖励 (给予方向感)
    if last_distance_to_goal is not None and current_distance_to_goal is not None:
        distance_change = last_distance_to_goal - current_distance_to_goal
        step_reward += distance_change * 1.5  # 1.5是可调整的权重
    else:
        # 如果没有距离信息，使用原有的距离奖励逻辑
        dist_reward = min(0.001, 0.05 * history_dist)
        step_reward += dist_reward

    # 4. 墙壁惩罚 (鼓励走在路中间)
    if map_info is not None:
        # 检查智能体中心周围的8个格子
        center_x, center_y = 5, 5  # 11x11视野的中心
        adjacent_cells = [
            (center_x-1, center_y-1), (center_x-1, center_y), (center_x-1, center_y+1),
            (center_x, center_y-1), (center_x, center_y+1),
            (center_x+1, center_y-1), (center_x+1, center_y), (center_x+1, center_y+1)
        ]
        wall_count = 0
        
        # 处理map_info的数据结构
        # map_info是一个包含字典的列表，每个字典有'values'键
        try:
            for x, y in adjacent_cells:
                if 0 <= x < 11 and 0 <= y < 11:
                    # 根据错误信息，map_info是包含字典的列表
                    if isinstance(map_info, list) and len(map_info) > x:
                        if isinstance(map_info[x], dict) and 'values' in map_info[x]:
                            cell_value = map_info[x]['values'][y] if y < len(map_info[x]['values']) else 1
                        else:
                            cell_value = map_info[x][y] if isinstance(map_info[x], (list, tuple)) else 1
                    else:
                        cell_value = 1  # 默认可通行
                    
                    if cell_value == 0:  # 0 代表墙壁/不可通行
                        wall_count += 1
        except Exception as e:
            # 如果处理地图信息出错，跳过墙壁惩罚
            pass
        
        step_reward -= wall_count * 0.01  # 0.01是可调整的权重

    # 5. 到达终点奖励
    if is_goal_reached:
        step_reward += 1000

    # 6. 收集宝箱奖励 (策略一：动态价值宝箱)
    if is_treasure_collected and remaining_steps is not None and max_steps is not None:
        # 宝箱的价值与剩余时间成正比
        dynamic_treasure_reward = 100 * (remaining_steps / max_steps)
        step_reward += dynamic_treasure_reward

    return [step_reward]


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


@attached
def SampleData2NumpyData(g_data):
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    obs_data_size = Config.DIM_OF_OBSERVATION
    legal_data_size = Config.DIM_OF_TOTAL_ACTION
    return SampleData(
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[2 * obs_data_size : 2 * obs_data_size + legal_data_size],
        _obs_legal=s_data[2 * obs_data_size + legal_data_size : 2 * obs_data_size + 2 * legal_data_size],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
