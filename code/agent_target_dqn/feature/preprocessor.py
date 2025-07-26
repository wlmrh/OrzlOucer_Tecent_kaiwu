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
        self.move_action_num = 8
        self.reset()
        self.visited_positions = set()  # 新增：记录访问过的位置
        self.position_tolerance = 1e-5  # 新增：位置判重的容差值

    def reset(self):
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = []
        self.bad_move_ids = set()

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

        # 获取当前位置（假设self.cur_pos_norm包含当前位置信息）
        current_position = tuple(np.round(self.cur_pos_norm, 5))  # 四舍五入到小数点5位
    
        # 检查位置重复
        repeated = False
        if hasattr(self, 'visited_positions'):
            # 检查是否访问过相同位置（考虑容差）
            for pos in self.visited_positions:
                if np.linalg.norm(np.array(current_position) - np.array(pos)) < self.position_tolerance:
                    repeated = True
                    break
        
            if not repeated:
                self.visited_positions.add(current_position)
    
        # 原有处理流程
        legal_action = self.get_legal_action()
        feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, 
                                 self.feature_history_pos, legal_action])
    
        # 计算基础奖励
        base_reward = reward_process(self.feature_end_pos[-1], self.feature_history_pos[-1])[0]

        # 添加位置重复惩罚
        if repeated:
            repeated_penalty = -0.05  # 每次重复访问的惩罚值
            print(f"重复位置惩罚! 位置: {current_position}, 惩罚: {repeated_penalty}")
            final_reward = base_reward + repeated_penalty
        else:
            final_reward = base_reward
    
        return (feature, legal_action, [final_reward])

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

        legal_action = [self.move_usable] * self.move_action_num
        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        if self.move_usable not in legal_action:
            self.bad_move_ids = set()
            return [self.move_usable] * self.move_action_num

        return legal_action
