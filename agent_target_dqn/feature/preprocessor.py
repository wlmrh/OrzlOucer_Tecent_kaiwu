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
from agent_target_dqn.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process


def norm(v, max_v, min_v=0):  # 将值 v 归一化到 [0, 1] 范围内
    v = np.maximum(np.minimum(max_v, v), min_v) # 将 v 限制在 [min_v, max_v] 范围内
    return (v - min_v) / (max_v - min_v)


class Preprocessor: # 该类存储并处理游戏相关的状态信息
    def __init__(self) -> None:
        self.move_action_num = 16
        self.reset()

    def reset(self):
        self.step_no = 0 # 当前步数
        self.cur_pos = (0, 0) # 当前位置
        self.cur_pos_norm = np.array((0, 0)) # 当前方向
        self.end_pos = None # 终点位置
        self.is_end_pos_found = False # 是否找到终点位置
        self.history_pos = [] # 曾到过的位置
        self.bad_move_ids = set() # 不可用的移动动作

    # 参数分别为终点是否已经被找到、当前位置和认为的终点位置（如果已经找到则为真实位置，否则为预测位置）
    def _get_pos_feature(self, found, cur_pos, target_pos):
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos)) # 终点位置相对于起点位置的坐标
        dist = np.linalg.norm(relative_pos) # 计算当前位置和终点位置的直线距离
        target_pos_norm = norm(target_pos, 128, -128) # 归一化后的终点坐标
        feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1), # 将当前位置相对于终点位置的 x 坐标归一化到 [-1, 1] 范围
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1), 
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * 128), # 将当前位置和终点位置的距离归一化到 [0, 1] 范围(0 ~ 128 * sqrt(2))
            ),
        )
        return feature

    def pb2struct(self, frame_state, last_action):
        obs, _ = frame_state
        self.step_no = obs["frame_state"]["step_no"]

        hero = obs["frame_state"]["heroes"][0] # 获取英雄id
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        # History position
        # 历史位置
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        # End position
        # 终点位置
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4:
                end_pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
                end_pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
                if organ["status"] != -1:
                    self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                    self.is_end_pos_found = True
                # if end_pos is not found, use relative position to predict end_pos
                # 如果终点位置未找到，使用相对位置预测终点位置
                elif (not self.is_end_pos_found) and (
                    self.end_pos is None
                    or self.step_no % 100 == 0
                    or self.end_pos_dir != end_pos_dir
                    or self.end_pos_dis != end_pos_dis
                ):
                    distance = end_pos_dis * 20
                    theta = DirectionAngles[end_pos_dir]
                    delta_x = distance * math.cos(math.radians(theta))
                    delta_z = distance * math.sin(math.radians(theta))

                    self.end_pos = (
                        max(0, min(128, round(self.cur_pos[0] + delta_x))),
                        max(0, min(128, round(self.cur_pos[1] + delta_z))),
                    )

                    self.end_pos_dir = end_pos_dir
                    self.end_pos_dis = end_pos_dis

        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

        # History position feature
        # 历史位置特征
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

        self.move_usable = True
        self.last_action = last_action

    def process(self, frame_state, last_action):
        self.pb2struct(frame_state, last_action)

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()

        # Feature
        # 特征
        feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, self.feature_history_pos, legal_action])

        return (
            feature,
            legal_action,
            # 第一个参数为当前位置到终点的归一化后的距离，第二个参数为当前位置到起始位置的归一化后的距离
            reward_process(self.feature_end_pos[-1], self.feature_history_pos[-1])
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

        legal_action = [self.move_usable] * self.move_action_num
        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        if self.move_usable not in legal_action:
            self.bad_move_ids = set()
            return [self.move_usable] * self.move_action_num

        return legal_action

