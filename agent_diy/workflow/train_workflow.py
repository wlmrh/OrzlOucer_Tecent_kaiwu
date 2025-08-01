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
from kaiwu_agent.utils.common_func import Frame, attached
from agent_diy.feature.definition import ReplayBuffer
from tools.train_env_conf_validate import read_usr_conf
from agent_diy.feature.definition import (
    sample_process,
)
from tools.metrics_utils import get_training_metrics
from agent_diy.conf.conf import Config

@attached
def workflow(envs, agents, logger=None, monitor=None):
    try:
        env, agent = envs[0], agents[0]
        episode_num_every_epoch = 1
        last_save_model_time = 0
        last_put_data_time = 0
        monitor_data = {}
        # Read and validate configuration file
        # 配置文件读取和校验
        usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error(f"usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
            return

        replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE) # 假设 Config 中有 REPLAY_BUFFER_SIZE
        # 预训练步数 (Pre-fill steps): 在开始学习前，先填充 Replay Buffer 到一定容量
        logger.info(f"Starting to pre-fill replay buffer up to {Config.REPLAY_BUFFER_MIN_SIZE} experiences...")
        while replay_buffer.size() < Config.REPLAY_BUFFER_MIN_SIZE:
            for g_data, _ in run_episodes(1, env, agent, usr_conf, logger, monitor):
                for sample in g_data:
                    replay_buffer.add(sample)
        logger.info(f"Replay buffer pre-filled with {replay_buffer.size()} experiences. Starting training.")

        while True:
            for g_data, monitor_data in run_episodes(episode_num_every_epoch, env, agent, usr_conf, logger, monitor):
                # 传入 replay_buffer
                for samples in g_data:
                    replay_buffer.add(samples)
                if replay_buffer.size() >= Config.REPLAY_BUFFER_MIN_SIZE:
                    for _ in range(Config.TRAIN_ITERATIONS_PER_EPISODE): # 假设每个回合训练多次
                        if replay_buffer.size() < Config.BATCH_SIZE: # 防止 batch size 大于 buffer size
                            break
                        
                        sampled_batch = replay_buffer.sample(Config.BATCH_SIZE)
                        agent.learn(sampled_batch)

            # Save model file
            # 保存model文件
            now = time.time()
            if now - last_save_model_time >= 1800: # 30min 保存一次模型
                agent.save_model()
                last_save_model_time = now

            # Report monitoring metrics
            # 上报监控指标
            if now - last_put_data_time >= 60: # 1min 更新一次 monitor
                monitor.put_data({os.getpid(): monitor_data})
                last_put_data_time = now

    # except Exception as e:
    #     raise RuntimeError(f"workflow error")
    except Exception as e:
        logger.exception("workflow error")
        raise RuntimeError("workflow error") from e


def run_episodes(n_episode, env, agent, usr_conf, logger, monitor):
    try:
        for episode in range(n_episode):
            collector = list()
            win_rate = 0

            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                logger.info(f"training_metrics is {training_metrics}")

            # Reset the task and get the initial state
            # 重置任务, 并获取初始状态
            obs, extra_info = env.reset(usr_conf=usr_conf)
            if extra_info["result_code"] < 0:
                logger.error(
                    f"env.reset result_code is {extra_info['result_code']}, result_message is {extra_info['result_message']}"
                )
                raise RuntimeError(extra_info["result_message"])
            elif extra_info["result_code"] > 0:
                continue

            # At the start of each game, support loading the latest model file
            # The call will load the latest model from a remote training node
            # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
            agent.reset()
            agent.load_model(id="latest")

            # Feature processing
            # 特征处理
            obs_data, _ = agent.observation_process(obs, extra_info) # 不需要奖励函数项

            done = False
            step = 0
            diy_1 = 0
            diy_2 = 0
            diy_3 = 0
            diy_4 = 0
            diy_5 = 0

            max_step_no = int(os.environ.get("max_step_no", "0"))

            while not done:
                # Agent performs inference, gets the predicted action for the next frame
                # Agent 进行推理, 获取下一帧的预测动作
                act_data, model_version = agent.predict(list_obs_data=[obs_data])

                # Unpack ActData into action
                # ActData 解包成动作
                act = agent.action_process(act_data[0])

                # Interact with the environment, execute actions, get the next state
                # 与环境交互, 执行动作, 获取下一步的状态
                step_no, _obs, terminated, truncated, _extra_info = env.step(act)
                if _extra_info["result_code"] != 0:
                    logger.warning(
                        f"_extra_info.result_code is {_extra_info['result_code']}, \
                        _extra_info.result_message is {_extra_info['result_message']}"
                    )
                    break

                step += 1

                # Feature processing
                # 特征处理
                _obs_data, reward_list = agent.observation_process(_obs, _extra_info, truncated)
                reward = sum(reward_list)

                # Determine task over, and update the number of victories
                # 判断任务结束, 并更新胜利次数
                game_info = _extra_info["game_info"]
                if truncated:
                    win_rate = agent.update_win_rate(False)
                    logger.info(
                        f"Game truncated! step_no:{step_no} score:{game_info['total_score']} win_rate:{win_rate}"
                    )
                elif terminated:
                    win_rate = agent.update_win_rate(True)
                    logger.info(
                        f"Game terminated! step_no:{step_no} score:{game_info['total_score']} win_rate:{win_rate}"
                    )
                done = terminated or truncated or (max_step_no > 0 and step >= max_step_no)

                # Construct task frames to prepare for sample construction
                # 构造任务帧，为构造样本做准备
                frame = Frame(
                    obs=obs_data.feature,
                    _obs=_obs_data.feature,
                    obs_legal=obs_data.legal_act,
                    _obs_legal=_obs_data.legal_act,
                    act=act,
                    rew=reward,
                    done=done,
                    ret=reward,
                )

                collector.append(frame)

                # If the task is over, the sample is processed and sent to training
                # 如果任务结束，则进行样本处理，将样本送去训练
                if done:
                    if monitor:
                        monitor_data = {
                            "diy_1": win_rate,
                            "diy_2": diy_2,
                            "diy_3": diy_3,
                            "diy_4": diy_4,
                            "diy_5": diy_5,
                        }

                    if len(collector) > 0:
                        collector = sample_process(collector)
                        yield collector, monitor_data # collector 是 sample_data 的 list
                    break

                # Status update
                # 状态更新
                obs_data = _obs_data
                extra_info = _extra_info
    except Exception as e:
        logger.exception("run_episodes error")
        raise RuntimeError("run_episodes error") from e

