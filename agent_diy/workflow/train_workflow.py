#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from agent_diy.feature.definition import (
    sample_process,
    reward_shaping,
)
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from tools.train_env_conf_validate import read_usr_conf
import time
import math
import os


@attached
def workflow(envs, agents, logger=None, monitor=None):
    """
    Users can define their own training workflows here
    用户可以在此处自行定义训练工作流
    """

    try:
        # Read and validate configuration file
        # 配置文件读取和校验
        usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
            return

        env, agent = envs[0], agents[0]
        EPISODES = 100
        SAVE_INTERVAL = 10
        TARGET_UPDATE_INTERVAL = 10
        BATCH_SIZE = 32
        REPORT_INERVAL = 60

        monitor_data = {"reward": 0, "loss": 0, "win_rate": 0}
        last_report_time = last_save_time = time.time()
        total_reward = total_loss = total_steps = win_cnt = 0

        logger.info("Start DQN Training...")
        start_time = time.time()

        for episode in range(EPISODES):
            obs, extra_info = env.reset(usr_conf=usr_conf)
            obs_data = agent.observation_process(obs, extra_info)
            if extra_info["result_code"] != 0:
                logger.error(
                    f"env.reset result_code is {extra_info['result_code']}, result_message is {extra_info['result_message']}"
                )
                raise RuntimeError(extra_info["result_message"])
            
            done = False # 当前游戏是否结束
            episode_reward = 0 # 当前轮的总奖励

            while not done:
                agent.epsilon = max(0.1, agent.epsilon * math.exp(-episode / EPISODES)) # 更新当前帧数的 epsilon
                act_data, _ = agent.predict([obs_data])
                action = agent.action_process(act_data[0])

                # 执行 action，获得下一步的状态和 reward
                frame_no, next_obs, terminated, truncated, next_extra_info = env.step(action)
                next_obs_data = agent.observation_process(next_obs, next_extra_info)

                # 需要根据具体的奖励函数进行修改
                reward = agent.reward_shaping(frame_no, next_extra_info)

                done = terminated or truncated
                if terminated:
                    win_cnt += 1

                episode_reward += reward

                # 生成训练所需要的 frame （需要修改）
                transition = Frame(
                    state=obs_data.feature,
                    action=action,
                    reward=reward,
                    next_state=next_obs_data.feature,
                )
                agent.replay_buffer.push(transition)

                if len(agent.replay_buffer) >= BATCH_SIZE:
                    batch = agent.replay_buffer.sample(BATCH_SIZE)
                    loss = agent.learn(batch)
                    total_loss += loss
                    total_steps += 1

                # 切换到下一个状态
                obs_data = next_obs_data

            # 累积统计
            total_reward += episode_reward

            # 定期更新目标网络
            if episode % TARGET_UPDATE_INTERVAL == 0:
                agent.update_target_network()

            # 定期保存模型
            now = time.time()
            if now - last_save_time > SAVE_INTERVAL:
                logger.info(f"Episode {episode}: saving model...")
                agent.save_model()
                last_save_time = now

            # 周期性上报训练进度
            is_converged = win_cnt / (episode + 1) > 0.9 and episode > 100
            if is_converged or now - last_report_time > REPORT_INERVAL:
                avg_reward = total_reward / episode
                avg_loss = total_loss / max(1, total_steps)
                win_rate = win_cnt / episode
                logger.info(f"Epi {episode}: avg_reward={avg_reward:.3f}, avg_loss={avg_loss:.4f}, win_rate={win_rate:.2%}")

                if monitor:
                    monitor.put_data({os.getpid(): monitor_data})

                monitor_data.update({"reward": 0, "loss": 0, "win_rate": 0})
                last_report_time = now

        total_time = time.time() - start_time
        agent.episodes = episode + 1
        logger.info(f"Training Time for {episode} episodes: {total_time:.2f} s")
        agent.save_model()

    except Exception as e:
        raise RuntimeError(f"workflow error")
