#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import math
from agent_diy.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process


def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)

# 0表示不可通行，1表示可以通行，2表示起点位置，3表示终点位置，4表示宝箱位置，6表示加速增益位置
class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 16
        self.reset()

    def reset(self):
        self.step_no = 0 # 当前步数
        self.cur_pos = (0, 0) # 当前位置
        self.cur_pos_norm = np.array((0, 0)) # 当前归一化坐标
        self.end_pos = None # 终点位置
        self.end_pos_norm = None # 归一化之后的终点坐标
        self.is_end_pos_found = False # 记录终点位置是否已经被找到
        self.end_dis = 1000.0 # 当前到终点的距离
        self.pri_end_dis = 1000.0 # 上一帧到终点的距离
        self.bad_move_ids = set() # 用来处理在当前位置，哪些操作是非法的
        self.can_flash = True # 闪现是否能使用
        self.near_treasure = None # 最近宝箱坐标
        self.near_treasure_norm = None # 最近宝箱归一化坐标
        self.near_treasure_dis = 1000.0 # 最近宝箱距离
        self.get_treature = 0 # 已收集的宝箱数量
        self.pri_near_treasure_dis = 1000.0 # 上一帧到最近宝箱的距离
        self.vision = [np.zeros((11, 11), dtype=int) for i in range(5)]
        # 通道 0: 正常道路 (1) / 非正常道路 (0)
        # 通道 1: 障碍物 (1) / 非障碍物 (0)
        # 通道 2: 宝箱 (1) / 非宝箱 (0)
        # 通道 3: 终点 (1) / 非终点 (0)
        # 通道 4: 加速增益位置 (1) / 非加速增益位置 (0)

    def _get_pos_feature(self, found, cur_pos, target_pos): # 暂时没用
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm(target_pos, 128, -128)
        feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * 128),
            ),
        )
        return feature

    def get_approx_loc(pos_dis, pos_dir): # 根据相对大致方位，大致距离，来计算物件的大致相对坐标
        distance = pos_dis * 20
        theta = DirectionAngles[pos_dir]
        delta_x = distance * math.cos(math.radians(theta))
        delta_z = distance * math.sin(math.radians(theta))
        return delta_x, delta_z

    def pb2struct(self, frame_state, last_action):
        obs, _ = frame_state
        self.step_no += 1

        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        self.cur_pos_norm = norm(self.cur_pos, 128, 0)

        if hero['talent']['status'] == 0:
            self.can_flash = False
        else:
            self.can_flash = True

        # End and Treasure position
        # 终点和最近宝箱位置
        self.pri_near_treasure_dis = self.near_treasure_dis
        self.pri_end_dis = self.end_dis
        self.near_treasure_dis = 1000.0
        self.get_treasure = 0
        for organ in obs["frame_state"]["organs"]:
            pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
            pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
            if organ["sub_type"] == 3: # 如果是终点
                if organ["status"] != -1: # 0表示不可获取，1表示可获取, -1表示视野外
                    self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                    self.end_pos_norm = norm(self.end_pos, 128, 0)
                    self.end_dis = math.hypot(self.end_pos[0] - self.cur_pos[0], self.end_pos[1] - self.cur_pos[1])
                    self.is_end_pos_found = True
                # if end_pos is not found, use relative position to predict end_pos
                # 如果终点位置未找到，使用相对位置预测终点位置
                elif (not self.is_end_pos_found) and ( # 如果曾在视野内，也不需要更改其坐标
                    self.end_pos is None
                    or self.end_pos_dir != end_pos_dir
                    or self.end_pos_dis != end_pos_dis
                ):
                    delta_x, delta_z = get_approx_loc(pos_dis, pos_dir)
                    self.end_pos = (
                        max(0, min(128, round(self.cur_pos[0] + delta_x))),
                        max(0, min(128, round(self.cur_pos[1] + delta_z))),
                    )
                    self.end_pos_norm = norm(self.end_pos, 128, 0)
                    self.end_dis = math.hypot(self.end_pos[0] - self.cur_pos[0], self.end_pos[1] - self.cur_pos[1])
            elif organ["sub_type"] == 4: # 如果是宝箱
                if organ["status"] == 0: # 已被获取（空宝箱）
                    self.get_treasure += 1
                    continue
                if organ["status"] != -1: # 在视野内，不需要估计大致坐标
                    loc = (organ["pos"]["x"], organ["pos"]["z"])
                    distance = math.hypot(loc[0] - self.near_treasure[0], loc[1] - self.near_treasure[1])
                    if distance < self.near_treasure_dis:
                        self.near_treasure = loc
                        self.near_treasure_norm = norm(near_treasure, 128, 0)
                        self.near_treasure_dis = distance
                # 不在视野内则估算距离
                else:
                    delta_x, delta_z = get_approx_loc(pos_dis, pos_dir)
                    loc = (
                        max(0, min(128, round(self.cur_pos[0] + delta_x))),
                        max(0, min(128, round(self.cur_pos[1] + delta_z))),
                    )
                    distance = math.hypot(loc[0] - self.near_treasure[0], loc[1] - self.near_treasure[1])
                    if distance < self.near_treasure_dis:
                        self.near_treasure = loc
                        self.near_treasure_norm = norm(near_treasure, 128, 0)
                        self.near_treasure_dis = distance
        
        # 更新视野
        # 0表示不可通行，1表示可以通行，2表示起点位置，3表示终点位置，4表示宝箱位置，6表示加速增益位置
        # vision
        # 通道 0: 正常道路 (1) / 非正常道路 (0)
        # 通道 1: 障碍物 (1) / 非障碍物 (0)
        # 通道 2: 宝箱 (1) / 非宝箱 (0)
        # 通道 3: 终点 (1) / 非终点 (0)
        # 通道 4: 加速增益位置 (1) / 非加速增益位置 (0)
        for r, row_data in enumerate(obs["map_info"]):
            for c, value in enumerate(row_data['values']):
                if value == 0:
                    vision[1][r, c] = 1
                elif value == 1 or value == 2:
                    vision[0][r, c] = 1
                elif value == 3:
                    vision[3][r, c] = 1
                elif value == 4:
                    vision[2][r, c] = 1
                else:
                    vision[4][r, c] = 1

        self.last_action = last_action

    def process(self, frame_state, last_action):
        self.pb2struct(frame_state, last_action) # 更新 Preprocessor 类的各项属性

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()

        # (5, 11, 11) 展平后的视野
        local_grid_flat = torch.from_numpy(self.vision.astype(np.float32)).view(-1)

        # 游戏数据
        game_info = frame_state["extra_info"]["game_info"]

        # 随机障碍物编号独热编码
        obstacle = [0] * 6
        obstacle[game_info["obstacle_id"]] = 1

        # Feature
        # 特征(设计见 conf.py/FEATURES)
        feature = np.concatenate([local_grid_flat, [self.can_flash], [self.get_treasure], [self.step_no],
        self.cur_pos_norm, self.near_treasure_norm, self.end_pos_norm, obstacle])

        return (
            feature,
            legal_action,
            reward_process(game_info["score"], self.cur_pos, self.near_treasure, self.end_pos,
                   self.pri_near_treasure_dis, self.pri_end_dis, self.step_no, (game_info["pos"] == game_info["end_pos"]),
                   (legal_action[self.last_action] == 0))
        )

    def get_legal_action(self):
        if not self.can_flash:
            for idx in range(8, 16):
                self.bad_move_ids.add(idx)
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

        legal_action = [True] * self.move_action_num
        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        # if self.move_usable not in legal_action:
        #     self.bad_move_ids = set()
        #     return [self.move_usable] * self.move_action_num

        return legal_action
