#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import torch
import math
from agent_diy.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process

# Assuming Config is available for map dimensions, etc.
# from kaiwudrl.common.config.config_control import CONFIG # If using KaiwuDRL's Config

def norm(v, max_v, min_v=0):
    """
    归一化函数，将值v归一化到[0, 1]区间。
    v可以是标量或NumPy数组。
    """
    # 确保 v 是 NumPy 数组，以便进行元素级操作
    v = np.asarray(v, dtype=np.float32)
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)

# 0表示不可通行，1表示可以通行，2表示起点位置，3表示终点位置，4表示宝箱位置，6表示加速增益位置
class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 16 # 总动作数量，包括移动和闪现
        self.map_size = 128 # 假设地图大小为 128x128，用于坐标归一化
        self.reset()

    def reset(self):
        self.step_no = 0 # 当前步数
        self.cur_pos = None # 当前位置
        self.pri_pos = None # 上一帧的位置
        # 确保这些坐标始终是 np.array，并有默认值
        self.cur_pos_norm = np.array([0.0, 0.0], dtype=np.float32) # 当前归一化坐标
        
        self.end_pos = None # 终点位置
        self.end_pos_norm = np.array([-1.0, -1.0], dtype=np.float32) # 归一化之后的终点坐标，-1.0表示未找到
        self.is_end_pos_found = False # 记录终点位置是否已经被找到
        self.end_dis = 1000.0 # 当前到终点的距离
        self.pri_end_dis = 1000.0 # 上一帧到终点的距离
        
        self.bad_move_ids = set() # 用来处理在当前位置，哪些操作是非法的
        self.can_flash = True # 闪现是否能使用
        
        self.near_treasure = None # 最近宝箱坐标
        self.near_treasure_norm = np.array([-1.0, -1.0], dtype=np.float32) # 最近宝箱归一化坐标，-1.0表示未找到
        self.near_treasure_dis = 1000.0 # 最近宝箱距离
        self.get_treasure = 0 # 已收集的宝箱数量
        self.pri_near_treasure_dis = 1000.0 # 上一帧到最近宝箱的距离
        
        # 视觉信息初始化为 float32
        self.vision = [np.zeros((11, 11), dtype=np.float32) for _ in range(5)]
        # 通道 0: 正常道路 (1) / 非正常道路 (0)
        # 通道 1: 障碍物 (1) / 非障碍物 (0)
        # 通道 2: 宝箱 (1) / 非宝箱 (0)
        # 通道 3: 终点 (1) / 非终点 (0)
        # 通道 4: 加速增益位置 (1) / 非加速增益位置 (0)

        # 用于 get_legal_action 的上一帧位置
        self.last_pos_norm = np.array([0.0, 0.0], dtype=np.float32) 
        self.last_action = -1 # 上一帧的动作

    def _get_pos_feature(self, found, cur_pos, target_pos): # 暂时没用
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm(target_pos, self.map_size, 0) # 使用 self.map_size
        feature = np.array(
            (
                float(found), # 转换为浮点数
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * self.map_size), # 使用 self.map_size
            ),
            dtype=np.float32 # 确保数组类型
        )
        return feature

    def get_approx_loc(self, pos_dis, pos_dir): # 根据相对大致方位，大致距离，来计算物件的大致相对坐标
        distance = pos_dis * 20
        theta = DirectionAngles[pos_dir]
        delta_x = distance * math.cos(math.radians(theta))
        delta_z = distance * math.sin(math.radians(theta))
        return delta_x, delta_z

    def pb2struct(self, frame_state, last_action):
        obs, _ = frame_state
        self.step_no += 1

        hero = obs["frame_state"]["heroes"][0]
        self.pri_pos = self.cur_pos
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        self.last_pos_norm = self.cur_pos_norm.copy() # 拷贝旧的归一化位置
        self.cur_pos_norm = norm(self.cur_pos, self.map_size, 0) # 使用 self.map_size

        if hero['talent']['status'] == 0:
            self.can_flash = False
        else:
            self.can_flash = True

        # End and Treasure position
        self.pri_near_treasure_dis = self.near_treasure_dis
        self.pri_end_dis = self.end_dis
        
        # 重置最近宝箱和终点信息，以便重新发现
        self.near_treasure_dis = 1000.0
        self.near_treasure = None
        self.near_treasure_norm = np.array([-1.0, -1.0], dtype=np.float32)

        # 终点位置的处理，如果视野中没有，则使用默认的未找到值
        if not self.is_end_pos_found:
            self.end_pos = None
            self.end_pos_norm = np.array([-1.0, -1.0], dtype=np.float32)
            self.end_dis = 1000.0

        self.get_treasure = 0 # 每次清零，重新计数当前帧收集到的宝箱

        for organ in obs["frame_state"]["organs"]:
            pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
            pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
            
            if organ["sub_type"] == 3: # 如果是终点
                if organ["status"] != -1: # 0表示不可获取，1表示可获取, -1表示视野外
                    self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                    self.end_pos_norm = norm(self.end_pos, self.map_size, 0) # 确保归一化
                    self.end_dis = math.hypot(self.end_pos[0] - self.cur_pos[0], self.end_pos[1] - self.cur_pos[1])
                    self.is_end_pos_found = True
                # if end_pos is not found, use relative position to predict end_pos
                # 如果终点位置未找到，使用相对位置预测终点位置
                elif (not self.is_end_pos_found): # 如果未曾找到过终点，且当前帧终点不在视野内，则进行估计
                    # 注意：你这里使用 end_pos_dir 和 end_pos_dis 作为条件，但这两个变量并未在 Preprocessor 类中定义为属性，
                    # 它们可能是在之前某个作用域中的局部变量，或者你想用 pos_dir 和 pos_dis。
                    # 为了避免 undefined 错误，我暂时去掉了这些条件。
                    # 如果你确实有这些变量，请确保它们是类属性并在适当位置更新。
                    
                    # 只有在 self.end_pos 还是 None 且 self.is_end_pos_found 为 False 时才估算
                    if self.end_pos is None: # 避免重复估算，只在没找到确切位置时估算一次
                        delta_x, delta_z = self.get_approx_loc(pos_dis, pos_dir)
                        self.end_pos = (
                            max(0, min(self.map_size -1, round(self.cur_pos[0] + delta_x))), # 确保在地图范围内
                            max(0, min(self.map_size -1, round(self.cur_pos[1] + delta_z))),
                        )
                        self.end_pos_norm = norm(self.end_pos, self.map_size, 0) # 确保归一化
                        self.end_dis = math.hypot(self.end_pos[0] - self.cur_pos[0], self.end_pos[1] - self.cur_pos[1])
            
            elif organ["sub_type"] == 4: # 如果是宝箱
                if organ["status"] == 0: # 已被获取（空宝箱）
                    self.get_treasure += 1
                    continue
                
                loc = (organ["pos"]["x"], organ["pos"]["z"])
                distance_to_current_organ = math.hypot(loc[0] - self.cur_pos[0], loc[1] - self.cur_pos[1])
                
                if organ["status"] != -1: # 在视野内，使用精确坐标
                    if distance_to_current_organ < self.near_treasure_dis:
                        self.near_treasure = loc
                        self.near_treasure_norm = norm(loc, self.map_size, 0) # 确保归一化
                        self.near_treasure_dis = distance_to_current_organ
                else: # 不在视野内则估算距离
                    delta_x, delta_z = self.get_approx_loc(pos_dis, pos_dir)
                    estimated_loc = (
                        max(0, min(self.map_size -1, round(self.cur_pos[0] + delta_x))), # 确保在地图范围内
                        max(0, min(self.map_size -1, round(self.cur_pos[1] + delta_z))),
                    )
                    distance_to_estimated_organ = math.hypot(estimated_loc[0] - self.cur_pos[0], estimated_loc[1] - self.cur_pos[1])

                    if distance_to_estimated_organ < self.near_treasure_dis:
                        self.near_treasure = estimated_loc
                        self.near_treasure_norm = norm(estimated_loc, self.map_size, 0) # 确保归一化
                        self.near_treasure_dis = distance_to_estimated_organ
        
        # 更新视野 (vision)
        # 清空所有通道以填充新数据
        for i in range(5):
            self.vision[i].fill(0) # 将所有元素设为0

        for r, row_data in enumerate(obs["map_info"]):
            for c, value in enumerate(row_data['values']):
                # 确保 r, c 在 11x11 范围内
                if 0 <= r < 11 and 0 <= c < 11:
                    if value == 0: # 不可通行 (障碍物)
                        self.vision[1][r, c] = 1.0 # 障碍物通道
                    elif value == 1 or value == 2: # 正常道路 / 起点位置
                        self.vision[0][r, c] = 1.0 # 正常道路通道
                    elif value == 3: # 终点位置
                        self.vision[3][r, c] = 1.0 # 终点通道
                    elif value == 4: # 宝箱位置
                        self.vision[2][r, c] = 1.0 # 宝箱通道
                    else: # 加速增益位置 (假设 value == 6 或其他值)
                        self.vision[4][r, c] = 1.0 # 加速增益通道

        self.last_action = last_action

    def process(self, frame_state, last_action, truncated):
        self.pb2struct(frame_state, last_action) # 更新 Preprocessor 类的各项属性

        # Legal action
        legal_action = self.get_legal_action() # 返回的是一个 [True, False, ...] 列表

        # (5, 11, 11) 展平后的视野，dtype 已经在初始化时设定
        vision_flat = np.array(self.vision, dtype=np.float32).flatten()
        
        # 游戏数据
        game_info = frame_state[1]
        game_info = game_info["game_info"] if game_info is not None else None

        # Feature (确保所有元素都是 float32 类型的 NumPy 数组或标量)
        feature_list = [
            vision_flat,
            np.array([float(self.can_flash)], dtype=np.float32), # 布尔值转浮点
            np.array([float(self.get_treasure)], dtype=np.float32), # 整数转浮点
            np.array([float(self.step_no)], dtype=np.float32), # 整数转浮点
            self.cur_pos_norm, # 已经是 np.float32 数组
            self.near_treasure_norm, # 已经是 np.float32 数组，包含填充值
            self.end_pos_norm # 已经是 np.float32 数组，包含填充值
        ]
        
        # 验证每个元素的类型和形状
        # for i, item in enumerate(feature_list):
        #    print(f"Feature item {i}: type={type(item)}, dtype={item.dtype if isinstance(item, np.ndarray) else 'N/A'}, shape={item.shape if isinstance(item, np.ndarray) else 'N/A'}")

        feature = np.concatenate(feature_list).astype(np.float32)

        # 检查最终 feature 的维度是否与 Config.DIM_OF_OBSERVATION 匹配
        # print(f"Final feature shape: {feature.shape}, dtype: {feature.dtype}")

        processed_reward = reward_process(
            raw_reward=game_info["score"] if game_info is not None else 0, # 假设 game_info["score"] 是原始奖励
            agent_pos=self.cur_pos,
            prev_pos=self.pri_pos,
            nearest_treasure_pos=self.near_treasure,
            end_pos=self.end_pos,
            prev_dist_to_treasure=self.pri_near_treasure_dis,
            prev_dist_to_end=self.pri_end_dis,
            current_steps=self.step_no,
            is_terminal=((self.cur_pos == self.end_pos) or truncated == 1),
            is_bad_action=(self.last_action != -1 and not legal_action[self.last_action]),
            is_flash_used=(self.last_action > 7),
            is_truncated=truncated
        )

        return (
            feature,
            legal_action,
            processed_reward
        )

    def get_legal_action(self):
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中
        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()

        if not self.can_flash:
            for idx in range(8, 16):
                self.bad_move_ids.add(idx)

        legal_action = [1] * self.move_action_num
        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0
        return legal_action