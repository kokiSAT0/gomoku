# -*- coding: utf-8 -*-
"""QAgent 同士で自己対戦を行う関数"""

from typing import Tuple, List, Optional
from tqdm import tqdm

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import QAgent


def selfplay_q_vs_q(
    board_size: int = 7,
    episodes: int = 500,
    env_params: Optional[dict] = None,
    black_params: Optional[dict] = None,
    white_params: Optional[dict] = None,
) -> Tuple[QAgent, QAgent, List[float], List[float], List[int], List[int]]:
    """QAgent 同士で自己対戦し学習を進める"""
    if env_params is None:
        env_params = {}
    if black_params is None:
        black_params = {}
    if white_params is None:
        white_params = {}

    # --- 初期化 ----------------------------------------------------------
    env = GomokuEnv(board_size=board_size, **env_params)
    black_agent = QAgent(board_size=board_size, **black_params)
    white_agent = QAgent(board_size=board_size, **white_params)

    all_rewards_black: List[float] = []
    all_rewards_white: List[float] = []
    winners: List[int] = []
    turns_list: List[int] = []

    for _ in tqdm(range(episodes), desc="Q vs Q"):
        obs = env.reset()
        done = False
        black_episode_reward = 0.0
        white_episode_reward = 0.0

        while not done:
            current_player = env.current_player
            state = obs.copy()
            if current_player == 1:
                action = black_agent.get_action(obs, env)
            else:
                action = white_agent.get_action(obs, env)

            next_obs, reward, done, info = env.step(action)
            obs = next_obs

            if current_player == 1:
                black_agent.record_transition(state, action, reward, next_obs, done)
                black_episode_reward += reward
            else:
                white_agent.record_transition(state, action, reward, next_obs, done)
                white_episode_reward += reward

        black_agent.finish_episode()
        white_agent.finish_episode()

        winners.append(info["winner"])
        turns_list.append(env.turn_count)
        all_rewards_black.append(black_episode_reward)
        all_rewards_white.append(white_episode_reward)

    return (
        black_agent,
        white_agent,
        all_rewards_black,
        all_rewards_white,
        winners,
        turns_list,
    )
