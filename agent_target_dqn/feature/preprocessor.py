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
        self.talent_action_num = 8
        self.total_action_num = 16  # 8个移动动作 + 8个闪现动作
        self.reset()

    def reset(self):
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = []
        self.bad_move_ids = set()
        self.last_distance_to_goal = None

    def _get_pos_feature(self, found, cur_pos, target_pos):
        try:
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
        except Exception as e:
            # 如果计算特征时出错，返回默认特征
            return np.array([found, 0, 0, 0.5, 0.5, 0.5])

    def pb2struct(self, frame_state, last_action):
        obs, _ = frame_state
        try:
            self.step_no = obs["frame_state"]["step_no"]
        except (KeyError, TypeError):
            self.step_no = 0

        try:
            hero = obs["frame_state"]["heroes"][0]
            self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        except (KeyError, TypeError, IndexError):
            self.cur_pos = (0, 0)

        # History position
        # 历史位置
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        # End position
        # 终点位置
        try:
            for organ in obs["frame_state"]["organs"]:
                if organ["sub_type"] == 4:
                    try:
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
                    except (KeyError, TypeError) as e:
                        # 如果无法获取终点信息，使用默认值
                        if self.end_pos is None:
                            self.end_pos = (64, 64)  # 默认终点位置
                        pass
        except (KeyError, TypeError):
            # 如果无法获取器官信息，使用默认终点位置
            if self.end_pos is None:
                self.end_pos = (64, 64)  # 默认终点位置

        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

        # History position feature
        # 历史位置特征
        if len(self.history_pos) > 0:
            self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])
        else:
            # 如果没有历史位置，使用当前位置
            self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.cur_pos)

        self.move_usable = True
        self.last_action = last_action

    def process(self, frame_state, last_action):
        self.pb2struct(frame_state, last_action)

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action(frame_state)

        # Feature
        # 特征
        try:
            feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, self.feature_history_pos, legal_action])
        except Exception as e:
            # 如果特征拼接出错，返回默认特征
            feature = np.concatenate([
                np.array([0.5, 0.5]),  # cur_pos_norm
                np.array([0, 0, 0, 0.5, 0.5, 0.5]),  # feature_end_pos
                np.array([0, 0, 0, 0.5, 0.5, 0.5]),  # feature_history_pos
                legal_action
            ])

        # 获取额外的奖励信息
        obs, extra_info = frame_state
        
        # 获取地图信息
        map_info = None
        try:
            if 'map_info' in obs:
                map_info = obs['map_info']
        except Exception:
            pass
        
        # 获取宝箱收集和终点到达信息
        is_treasure_collected = False
        is_goal_reached = False
        try:
            if 'frame_state' in obs:
                frame_state_data = obs['frame_state']
                # 检查是否有宝箱收集事件
                if 'events' in frame_state_data:
                    for event in frame_state_data['events']:
                        if event.get('type') == 'treasure_collected':
                            is_treasure_collected = True
                        elif event.get('type') == 'goal_reached':
                            is_goal_reached = True
        except Exception:
            pass
        
        # 获取步数信息
        remaining_steps = None
        max_steps = None
        try:
            if 'frame_state' in obs and 'step_no' in obs['frame_state']:
                current_step = obs['frame_state']['step_no']
                # 假设最大步数为2000，实际应该从配置中获取
                max_steps = 2000
                remaining_steps = max_steps - current_step
        except Exception:
            pass
        
        # 计算距离变化
        last_distance_to_goal = None
        current_distance_to_goal = None
        try:
            if hasattr(self, 'last_distance_to_goal'):
                last_distance_to_goal = self.last_distance_to_goal
            current_distance_to_goal = self.feature_end_pos[-1]
            self.last_distance_to_goal = current_distance_to_goal
        except Exception:
            pass

        try:
            reward_list = reward_process(
                self.feature_end_pos[-1], 
                self.feature_history_pos[-1],
                map_info=map_info,
                is_treasure_collected=is_treasure_collected,
                is_goal_reached=is_goal_reached,
                remaining_steps=remaining_steps,
                max_steps=max_steps,
                last_distance_to_goal=last_distance_to_goal,
                current_distance_to_goal=current_distance_to_goal
            )
        except Exception as e:
            # 如果奖励计算出错，返回默认奖励
            reward_list = [-0.1]  # 默认时间惩罚

        return (
            feature,
            legal_action,
            reward_list,
        )

    def get_legal_action(self, frame_state=None):
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

        # 初始化16维的合法动作列表
        legal_action = [1] * self.total_action_num  # 1表示可用，0表示不可用
        
        # 处理移动动作的合法性
        for move_id in self.bad_move_ids:
            if move_id < self.move_action_num:
                legal_action[move_id] = 0

        # 处理闪现动作的合法性
        # 从环境获取legal_act信息
        talent_available = 1  # 默认可用
        
        # 尝试从环境获取legal_act信息
        if frame_state is not None:
            obs, extra_info = frame_state
            if 'legal_act' in obs:
                legal_act = obs['legal_act']
                if isinstance(legal_act, list) and len(legal_act) >= 2:
                    # legal_act[1]表示超级闪现是否可用
                    talent_available = legal_act[1] if legal_act[1] in [0, 1] else 1
        
        # 设置闪现动作的合法性
        for i in range(self.move_action_num, self.total_action_num):
            legal_action[i] = talent_available

        return legal_action
