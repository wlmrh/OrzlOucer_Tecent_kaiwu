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


def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 16
        self.reset()

    def reset(self):
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = [] # 记录最近到过的10个位置
        self.bad_move_ids = set()
        self.can_flash = True
        self.graph = np.zeros((128, 128), dtype=int)
        self.discovery = 0
        self.dis_to_wall = 0
        self.discovery_pri = [1.0]
        self.is_getting_point = False
        for i in range(128):
            for j in range(128):
                self.graph[i, j] = -1

    def _get_pos_feature(self, found, cur_pos, target_pos):
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

    def pb2struct(self, frame_state, last_action):
        obs, _ = frame_state
        self.step_no = obs["frame_state"]["step_no"]

        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        if hero['talent']['status'] == 0:
            self.can_flash = False

        # Discovery new land
        # 探索到的新地块
        # 其中0表示不可通行，1表示可以通行，2表示起点位置，3表示终点位置，4表示宝箱位置，6表示加速增益位置。
        self.discovery = 0
        for r, row_data in enumerate(obs["map_info"]):
            for c, value in enumerate(row_data['values']):
                if (self.graph[r, c] == -1):
                    self.graph[r, c] = value
                    if value == 3:
                        self.discovery += 5
                    elif value == 1:
                        self.discovery += 1
                    elif value == 4:
                        self.discovery += 2
                    elif value == 6:
                        self.discovery += 3

        # Distance to the wall
        # 离墙距离
        crt_dis = 0
        for crt_dis in range(21):
            for x_dif in range(-crt_dis, crt_dis + 1):
                y_dif = crt_dis - abs(x_dif)
                crt_x = x_dif + self.cur_pos[0]
                crt_y = y_dif + self.cur_pos[1]
                if crt_x < 0 or crt_x >= 128 or crt_y < 0 or crt_y >= 128:
                    continue
                if self.graph[crt_x, crt_y] == 0:
                    break
        self.dis_to_wall = crt_dis

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
        feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, self.feature_history_pos, legal_action, [norm(self.dis_to_wall, 128, 0)]])

        return (
            feature,
            legal_action,
            reward_process(self.feature_end_pos[-1], self.feature_history_pos[-1], self.discovery, self.discovery_pri, self.dis_to_wall, self.step_no >= 700)
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
