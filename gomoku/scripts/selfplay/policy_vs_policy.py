# -*- coding: utf-8 -*-
"""PolicyAgent 同士で自己対戦を行う関数"""

from typing import Tuple, List, Optional
from tqdm import tqdm

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import PolicyAgent


def selfplay_policy_vs_policy(
    board_size: int = 7,
    episodes: int = 500,
    env_params: Optional[dict] = None,
    black_params: Optional[dict] = None,
    white_params: Optional[dict] = None,
) -> Tuple[PolicyAgent, PolicyAgent, List[float], List[float], List[int], List[int]]:
    """PolicyAgent 同士で自己対戦し学習を進める"""
    if env_params is None:
        env_params = {}
    if black_params is None:
        black_params = {}
    if white_params is None:
        white_params = {}

    # --- 環境とエージェントの初期化 -----------------------------------
    env = GomokuEnv(board_size=board_size, **env_params)
    black_agent = PolicyAgent(board_size=board_size, **black_params)
    white_agent = PolicyAgent(board_size=board_size, **white_params)

    # --- 学習結果の記録用リスト ---------------------------------------
    all_rewards_black: List[float] = []
    all_rewards_white: List[float] = []
    winners: List[int] = []
    turns_list: List[int] = []

    # --- 指定回数だけ自己対戦を繰り返す ------------------------------
    for _ in tqdm(range(episodes), desc="Policy vs Policy"):
        obs = env.reset()
        done = False
        black_episode_reward = 0.0
        white_episode_reward = 0.0

        while not done:
            current_player = env.current_player
            if current_player == 1:
                action = black_agent.get_action(obs, env)
            else:
                action = white_agent.get_action(obs, env)

            next_obs, reward, done, info = env.step(action)

            # 報酬の記録
            if current_player == 1:
                black_agent.record_reward(reward)
                black_episode_reward += reward
            else:
                white_agent.record_reward(reward)
                white_episode_reward += reward

            obs = next_obs

        # 1エピソード終了処理
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
