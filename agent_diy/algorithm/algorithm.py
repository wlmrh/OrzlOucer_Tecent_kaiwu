#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import os
import numpy as np
import torch
from copy import deepcopy
from agent_target_dqn.model.model import Model
from agent_target_dqn.conf.conf import Config
from agent_diy.feature.definition import ActData

好的，了解 sampled_batch 中每一项数据的具体格式 SampleData 后，我们需要对 Algorithm 类的 learn 方法进行相应的调整。

SampleData 结构如下：

obs: 当前状态的特征（s_t）

_obs: 下一状态的特征（s_t+1）

obs_legal: 当前状态的合法动作掩码

_obs_legal: 下一状态的合法动作掩码

act: 智能体在当前状态选择的动作 (a_t)

rew: 当前步获得的奖励 (r_t)

ret: (通常用于回合总回报，DQN中不直接使用)

done: 当前步是否导致回合终止

Algorithm 类 learn 方法的修改
主要的修改点在于 learn 方法中如何从 sampled_batch 中解包数据，以及确保将这些数据正确地转换为 PyTorch 张量并移动到指定设备。

Python

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import os
import numpy as np
import torch
from copy import deepcopy
from agent_target_dqn.model.model import Model
from agent_target_dqn.conf.conf import Config
# 假设 ActData 和 create_cls 都在这里或者你直接定义在这里
from agent_diy.feature.definition import ActData, create_cls # 确保 create_cls 被导入

# 定义 SampleData 类，如果它没有在其他地方导入的话
# from kaiwu_agent.agent.protocol import SampleData # 或者在你的 definition.py 中
# 如果 definition.py 中包含了 create_cls 的定义，那么 SampleData 应该会通过它创建。
# 假设 ReplayBuffer.sample 返回的是 SampleData 对象的列表
SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None, # 未使用
    done=None,
)


class Algorithm:
    def __init__(self, device, logger, monitor):
        # 动作空间维度
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION  # 8 个移动方向
        self.talent_direction = Config.DIM_OF_TALENT           # 8 个闪现方向
        self.total_action_space = Config.TOTAL_ACTION_SPACE    # 16 个总动作

        # 观测空间维度
        self.obs_shape = Config.DIM_OF_OBSERVATION             # 620 维特征

        # Epsilon-greedy 探索参数
        self.epsilon_max = Config.EPSILON_MAX
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY

        # DQN 相关参数
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR
        self.batch_size = Config.BATCH_SIZE # 从 Config 中获取 batch_size

        # 设备设置
        self.device = device

        # 模型初始化
        self.model = Model(
            state_shape=self.obs_shape,
            action_shape=self.total_action_space, # 动作输出维度是总动作空间
            softmax=False, # DQN 输出 Q 值，不需要 softmax
        )
        self.model.to(self.device)

        # 优化器
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Target 模型
        self.target_model = deepcopy(self.model)
        
        # 训练步数和预测计数
        self.train_step = 0
        self.predict_count = 0

        # 监控相关
        self.last_report_monitor_time = 0
        self.logger = logger
        self.monitor = monitor

    def learn(self, sampled_batch_list: list[SampleData]):
        """
        DQN 学习方法。
        Args:
            sampled_batch_list (list[SampleData]): 从 ReplayBuffer 采样出的批次数据列表。
                                                列表中的每个元素都是一个 SampleData 对象。
        """
        # 从 SampleData 对象列表中解包所有数据
        # obs: 当前状态特征
        states = [data.obs for data in sampled_batch_list]
        # _obs: 下一状态特征
        next_states = [data._obs for data in sampled_batch_list]
        # act: 动作索引
        actions = [data.act for data in sampled_batch_list]
        # rew: 奖励
        rewards = [data.rew for data in sampled_batch_list]
        # done: 结束标志
        dones = [data.done for data in sampled_batch_list]
        # _obs_legal: 下一状态的合法动作掩码
        next_legal_actions = [data._obs_legal for data in sampled_batch_list]

        # 转换为 PyTorch Tensor 并移动到设备
        batch_feature = self.__convert_to_tensor(states) # (BATCH_SIZE, OBS_DIM)
        batch_action = torch.LongTensor(actions).view(-1, 1).to(self.device) # (BATCH_SIZE, 1)
        batch_reward = torch.FloatTensor(rewards).to(self.device) # (BATCH_SIZE,)
        _batch_feature = self.__convert_to_tensor(next_states) # (BATCH_SIZE, OBS_DIM)
        
        # 将 done 标志转换为 not_done (0/1)
        # done=True -> not_done=0 (回合终止)
        # done=False -> not_done=1 (回合未终止)
        not_done = (~torch.BoolTensor(dones)).float().to(self.device) # (BATCH_SIZE,)
        
        # 下一状态的合法动作掩码，转换为布尔张量
        _batch_obs_legal = torch.BoolTensor(next_legal_actions).to(self.device) # (BATCH_SIZE, TOTAL_ACTION_SPACE)

        # 计算 Target Q 值
        self.target_model.eval() # 目标网络切换到评估模式
        with torch.no_grad():
            next_q_values = self.target_model(_batch_feature) # (BATCH_SIZE, TOTAL_ACTION_SPACE)

            # Mask 非法动作的 Q 值，将其设置为一个非常小的负数，确保不会被 max 选中
            next_q_values_masked = next_q_values.masked_fill(~_batch_obs_legal, float('-inf'))
            
            # 从合法动作中选择最大 Q 值 (Double DQN 会在这里使用主网络选择动作，目标网络评估Q值)
            next_q_max = next_q_values_masked.max(dim=1).values.detach() # (BATCH_SIZE,)

        # Q_target = reward + gamma * max_a' Q_target(s', a') * (1 - done)
        target_q = batch_reward + self._gamma * next_q_max * not_done # (BATCH_SIZE,)

        # 计算当前 Q 值 (Q_predict)
        self.optim.zero_grad()
        self.model.train() # 主网络切换到训练模式
        
        logits = self.model(batch_feature) # (BATCH_SIZE, TOTAL_ACTION_SPACE)
        
        # 提取实际执行动作的 Q 值
        q_predict = logits.gather(1, batch_action).view(-1) # (BATCH_SIZE,)

        # 计算 MSE Loss
        loss = torch.mean(torch.square(target_q - q_predict)) # (BATCH_SIZE,) 的均值

        # 反向传播和优化
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        model_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()

        self.train_step += 1

        # 更新 target 网络
        if self.train_step % self.target_update_freq == 0:
            self.update_target_q()
            self.logger.info(f"Target model updated at train step {self.train_step}")

        # 监控指标
        value_loss = loss.detach().item()
        q_value = q_predict.mean().detach().item() # 使用 q_predict 的平均值作为 Q 值监控
        reward = batch_reward.mean().detach().item() # 使用实际批量奖励的平均值

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
                "epsilon": self.epsilon # 监控 epsilon 值
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})
            self.last_report_monitor_time = now
            self.logger.info(
                f"Train Step: {self.train_step}, Loss: {value_loss:.4f}, Q_Value: {q_value:.2f}, Avg Reward: {reward:.2f}, Epsilon: {self.epsilon:.4f}"
            )

    def __convert_to_tensor(self, data):
        """
        辅助函数：将 NumPy 数组或列表转换为 PyTorch Tensor，并移动到指定设备。
        确保数据类型是 float32。
        """
        if isinstance(data, list):
            # 假设 list 中的元素是 NumPy 数组 (feature), 或者已经是 tensor
            if not data: # 处理空列表情况
                return torch.empty(0).to(self.device)
            
            if isinstance(data[0], torch.Tensor):
                processed = torch.stack(data, dim=0).to(self.device).float()
            elif isinstance(data[0], np.ndarray):
                # 如果是 numpy 数组，先堆叠再转 tensor
                processed = torch.from_numpy(np.stack(data, axis=0)).to(self.device).float()
            else:
                # 兼容其他类型，但推荐直接使用 np.ndarray 或 torch.Tensor
                processed = torch.tensor(np.array(data), dtype=torch.float32).to(self.device)
        elif isinstance(data, np.ndarray):
            # 直接从 numpy 数组转换
            processed = torch.from_numpy(data.astype(np.float32)).to(self.device)
        elif torch.is_tensor(data):
            # 如果已经是 tensor，直接移动和转换类型
            processed = data.to(self.device).float()
        else:
            raise TypeError(f"Unsupported data type for tensor conversion: {type(data)}")
        return processed

    def predict_detail(self, feature_tensor, legal_action_flags_list, exploit_flag=False):
        """
        智能体预测动作的方法。
        Args:
            feature_tensor (torch.Tensor): 预处理后的观测特征 (batch_size, obs_dim)。
                                           在 workflow/run_episodes 中，传入的是 (1, obs_dim) 用于单步预测。
            legal_action_flags_list (list of bool): 合法动作的布尔列表 (total_action_space)。
                                                 在 workflow/run_episodes 中，传入的是一个列表，需要转换为 tensor。
            exploit_flag (bool): 是否强制使用 Q 值进行开发 (不进行探索)。

        Returns:
            list[ActData]: 包含 move_dir 和 use_talent 信息的动作数据列表。
        """
        # predict_detail 通常处理一个 batch 的预测，即使 batch_size 为 1
        # 确保 feature_tensor 是 float32 并且在正确设备上 (在 __convert_to_tensor 已经处理)

        # legal_action_flags_list 传入的是一个列表，需要转换为 torch.BoolTensor
        # 这里假设 legal_action_flags_list 总是单例，即 list[list[bool]]，所以需要 legal_action_flags_list[0]
        # 或者在调用方确保传入的是一个 (total_action_space,) 的列表
        if isinstance(legal_action_flags_list[0], list): # 如果是 [[True, False, ...]]
            legal_act = torch.BoolTensor(legal_action_flags_list[0]).to(self.device)
        else: # 如果是 [True, False, ...]
            legal_act = torch.BoolTensor(legal_action_flags_list).to(self.device)


        # 如果是单步预测，需要添加 batch 维度 (batch_size=1)
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0) # (obs_dim,) -> (1, obs_dim)
            legal_act = legal_act.unsqueeze(0) # (total_action_space,) -> (1, total_action_space)

        # 探索因子更新
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
            -self.epsilon_decay * self.predict_count
        )

        self.model.eval() # 预测时切换到评估模式
        with torch.no_grad():
            # Epsilon-greedy 策略
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                # 随机选择合法动作
                random_values = torch.rand(feature_tensor.shape[0], self.total_action_space, device=self.device)
                
                # 将非法动作的随机值填充为非常小的数，确保它们不会被选中
                random_values_masked = random_values.masked_fill(~legal_act, float('-inf'))
                
                # 选择最大随机值对应的动作索引
                act_ids = random_values_masked.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                # 使用模型预测 Q 值
                logits = self.model(feature_tensor) # (batch_size, total_action_space)

                # 将非法动作的 Q 值填充为非常小的负数
                logits_masked = logits.masked_fill(~legal_act, float('-inf'))
                
                # 选择最大 Q 值对应的动作索引
                act_ids = logits_masked.argmax(dim=1).cpu().view(-1, 1).tolist()

        # 格式化动作输出
        # act_ids 是 [[action_index_0], [action_index_1], ...]
        # 需要将其转换为 (move_dir, use_talent_flag)
        format_action = []
        for instance_id_list in act_ids: # instance_id_list 应该是 [action_index]
            action_index = instance_id_list[0]
            move_dir = action_index % self.direction_space # 0-7 对应移动方向
            use_talent = 1 if action_index >= self.direction_space else 0 # 8-15 对应闪现动作
            format_action.append(ActData(move_dir=move_dir, use_talent=use_talent))
            
        self.predict_count += 1
        return format_action # 返回 ActData 列表

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())