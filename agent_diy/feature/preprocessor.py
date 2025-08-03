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
from agent_diy.conf.conf import Config
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

# 0表示不可通行，1表示可以通行，2表示起点坐标，3表示终点坐标，4表示宝箱坐标，6表示加速增益坐标
class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 16 # 总动作数量，包括移动和闪现
        self.map_size = 128 # 假设地图大小为 128x128，用于坐标归一化
        self.reset()

    def reset(self):
        self.current_steps = 0 # 当前步数
        self.agent_pos = None # 当前坐标
        self.agent_pos_norm = None # 归一化当前坐标
        self.prev_pos = None # 上一帧的坐标
        self.end_pos = None # 终点坐标
        self.end_pos_norm = None # 归一化终点坐标
        self.is_end_pos_precise = False # 记录终点坐标是否为准确坐标
        self.current_dist_to_end = float("inf") # 当前到终点的距离
        self.prev_dist_to_end = float("inf") # 上一帧到终点的距离
        self.bad_move_ids = set() # 记录非法操作
        self.can_flash = True # 闪现是否可用

        self.nearest_treasure_pos = None # 最近宝箱坐标
        self.nearest_treasure_pos_norm = None # 归一化最近宝箱坐标
        self.current_dist_to_treasure = float("inf") # 最近宝箱距离
        self.prev_dist_to_treasure = float("inf") # 上一帧到最近宝箱的距离
        self.treasures_got = [] # 已收集的宝箱编号
        self.is_getting_treasure = False # 当前帧是否获得宝箱
        
        # 视觉信息初始化为 float32
        self.vision = [np.zeros((11, 11), dtype=np.float32) for _ in range(5)]
        # 通道 0: 正常道路 (1) / 非正常道路 (0)
        # 通道 1: 障碍物 (1) / 非障碍物 (0)
        # 通道 2: 宝箱 (1) / 非宝箱 (0)
        # 通道 3: 终点 (1) / 非终点 (0)
        # 通道 4: 加速增益坐标 (1) / 非加速增益坐标 (0)

        # 用于 get_legal_action 的上一帧坐标
        self.last_pos_norm = None
        self.last_action = -1 # 上一帧的动作

    def get_approx_loc(self, pos_dis, pos_dir): # 根据相对大致方位，大致距离，来计算物件的大致相对坐标
        distance = pos_dis * 20
        theta = DirectionAngles[pos_dir]
        delta_x = distance * math.cos(math.radians(theta))
        delta_z = distance * math.sin(math.radians(theta))
        return delta_x, delta_z

    def pb2struct(self, frame_state, last_action):
        # 如果有障碍物，则不允许闪现
        self.can_flash = False

        obs, _ = frame_state
        self.current_steps += 1

        # 记录上一帧的信息
        self.prev_pos = self.agent_pos
        self.prev_dist_to_treasure = self.current_dist_to_treasure
        self.prev_dist_to_end = self.current_dist_to_end

        # 计算本帧的信息
        hero = obs["frame_state"]["heroes"][0]
        self.agent_pos = (hero["pos"]["x"], hero["pos"]["z"])
        self.last_pos_norm = self.agent_pos_norm
        self.agent_pos_norm = norm(self.agent_pos, self.map_size, 0)

        if hero['talent']['status'] == 0:
            self.can_flash = False
        elif hero['talent']['status'] == 1:
            self.can_flash = True
        
        # 重置最近宝箱距离，以便重新发现
        self.current_dist_to_treasure = float("inf")
        self.is_getting_treasure = False

        # 终点坐标的处理，如果视野中没有，则使用默认的未找到值
        if not self.is_end_pos_precise:
            self.end_pos = None
            self.end_pos_norm = None
            self.current_dist_to_end = float("inf")

        for organ in obs["frame_state"]["organs"]:
            pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
            pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
            delta_x, delta_z = self.get_approx_loc(pos_dis, pos_dir)

            if organ["sub_type"] == 4: # 如果是终点
                if self.is_end_pos_precise: # 曾经看到过终点的位置
                    self.current_dist_to_end = math.hypot(self.end_pos[0] - self.agent_pos[0], self.end_pos[1] - self.agent_pos[1])
                    continue

                if organ["status"] != -1: # 0表示不可获取，1表示可获取, -1表示视野外
                    self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                    self.end_pos_norm = norm(self.end_pos, self.map_size, 0) # 确保归一化
                    self.current_dist_to_end = math.hypot(self.end_pos[0] - self.agent_pos[0], self.end_pos[1] - self.agent_pos[1])
                    self.is_end_pos_precise = True
                # if end_pos is not found, use relative position to predict end_pos
                # 如果终点坐标未找到，使用相对坐标预测终点坐标
                elif (not self.is_end_pos_precise): # 终点一直在视野外
                    self.end_pos = (
                        max(0, min(self.map_size -1, round(self.agent_pos[0] + delta_x))), # 确保在地图范围内
                        max(0, min(self.map_size -1, round(self.agent_pos[1] + delta_z))),
                    )
                    self.end_pos_norm = norm(self.end_pos, self.map_size, 0) # 确保归一化
                    self.current_dist_to_end = math.hypot(self.end_pos[0] - self.agent_pos[0], self.end_pos[1] - self.agent_pos[1])
            
            elif organ["sub_type"] == 1: # 如果是宝箱
                if organ["status"] == 0: # 已被获取（空宝箱）
                    if orgain["config_id"] not in self.treasures_got:
                        self.treasures_got.append(orgain["config_id"])
                        self.is_getting_treasure = True
                    continue
                
                # 宝箱非空
                loc = (organ["pos"]["x"], organ["pos"]["z"])
                distance_to_current_organ = math.hypot(loc[0] - self.agent_pos[0], loc[1] - self.agent_pos[1])
                
                if organ["status"] != -1: # 在视野内，使用精确坐标
                    if distance_to_current_organ < self.current_dist_to_treasure:
                        self.nearest_treasure_pos = loc
                        self.nearest_treasure_pos_norm = norm(loc, self.map_size, 0) # 确保归一化
                        self.current_dist_to_treasure = distance_to_current_organ
                else: # 不在视野内则估算距离
                    estimated_loc = (
                        max(0, min(self.map_size -1, round(self.agent_pos[0] + delta_x))), # 确保在地图范围内
                        max(0, min(self.map_size -1, round(self.agent_pos[1] + delta_z))),
                    )
                    distance_to_estimated_organ = math.hypot(estimated_loc[0] - self.agent_pos[0], estimated_loc[1] - self.agent_pos[1])

                    if distance_to_estimated_organ < self.current_dist_to_treasure:
                        self.nearest_treasure_pos = estimated_loc
                        self.nearest_treasure_pos_norm = norm(estimated_loc, self.map_size, 0) # 确保归一化
                        self.current_dist_to_treasure = distance_to_estimated_organ
        
        # 更新视野 (vision)
        # 清空所有通道以填充新数据
        for i in range(5):
            self.vision[i].fill(0) # 将所有元素设为0

        # 0表示不可通行，1表示可以通行，2表示起点位置，3表示终点位置，4表示宝箱位置，6表示加速增益位置。
        for r, row_data in enumerate(obs["map_info"]):
            for c, value in enumerate(row_data['values']):
                # 确保 r, c 在 11x11 范围内
                if 0 <= r < 11 and 0 <= c < 11:
                    if value == 0: # 不可通行 (障碍物)
                        self.vision[1][r, c] = 1.0 # 障碍物通道
                    elif value == 1 or value == 2: # 正常道路 / 起点坐标
                        self.vision[0][r, c] = 1.0 # 正常道路通道
                    elif value == 3: # 终点坐标
                        self.vision[3][r, c] = 1.0 # 终点通道
                        self.vision[0][r, c] = 1.0 # 正常道路通道
                    elif value == 4: # 宝箱坐标
                        self.vision[2][r, c] = 1.0 # 宝箱通道
                        self.vision[0][r, c] = 1.0 # 正常道路通道
                    else: # 加速增益坐标 (假设 value == 6 或其他值)
                        self.vision[4][r, c] = 1.0 # 加速增益通道
                        self.vision[0][r, c] = 1.0 # 正常道路通道

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

        # 宝箱获得情况
        treasures_got_onehot = np.zeros(8, dtype=np.float32)
        for treasure_id in self.treasures_got:
            if 0 <= treasure_id < 8:
                treasures_got_onehot[treasure_id] = 1.0

        # Feature (确保所有元素都是 float32 类型的 NumPy 数组或标量)
        feature_list = [
            vision_flat,
            np.array([float(self.can_flash)], dtype=np.float32), # 布尔值转浮点
            np.array([float(len(self.treasures_got))], dtype=np.float32), # 整数转浮点
            treasures_got_onehot, # 已收集的宝箱编号，one-hot 编码
            np.array([float(self.current_steps / Config.MAX_STEP_NO)], dtype=np.float32), # 整数转浮点
            self.agent_pos_norm, # 已经是 np.float32 数组
            self.nearest_treasure_pos_norm, # 已经是 np.float32 数组，包含填充值
            self.end_pos_norm # 已经是 np.float32 数组，包含填充值
        ]
        
        # 验证每个元素的类型和形状
        # for i, item in enumerate(feature_list):
        #    print(f"Feature item {i}: type={type(item)}, dtype={item.dtype if isinstance(item, np.ndarray) else 'N/A'}, shape={item.shape if isinstance(item, np.ndarray) else 'N/A'}")

        feature = np.concatenate(feature_list).astype(np.float32)

        # 检查最终 feature 的维度是否与 Config.DIM_OF_OBSERVATION 匹配
        # print(f"Final feature shape: {feature.shape}, dtype: {feature.dtype}")

        processed_reward = reward_process(
            raw_reward=game_info["score"] if game_info is not None else 0,
            collected_treasures_count=len(self.treasures_got),
            agent_pos=self.agent_pos,
            prev_pos=self.prev_pos,
            nearest_treasure_pos=self.nearest_treasure_pos,
            end_pos=self.end_pos,
            current_dist_to_treasure=self.current_dist_to_treasure,
            prev_dist_to_treasure=self.prev_dist_to_treasure,
            current_dist_to_end=self.current_dist_to_end,
            prev_dist_to_end=self.prev_dist_to_end,
            current_steps=self.current_steps,
            is_terminal=((self.agent_pos == self.end_pos) or truncated == True or prev_pos is None), # 防止第一轮的reward被计算成 -inf
            is_bad_action=(self.last_action in self.bad_move_ids),
            is_flash_used=(self.last_action > 7),
            is_getting_treasure=self.is_getting_treasure
        )

        return (
            feature,
            legal_action,
            processed_reward
        )

    def get_legal_action(self):
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前坐标与上一步坐标相同，则将该动作加入到bad_move_ids中
        if (
            self.last_pos_norm is not None
            and self.last_action is not None
            and abs(self.agent_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.agent_pos_norm[1] - self.last_pos_norm[1]) < 0.001
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