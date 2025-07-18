# -*- coding: utf-8 -*-
"""PolicyAgent と QAgent の組み合わせ自己対戦用関数"""

from typing import Tuple, List, Optional
from tqdm import tqdm

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import PolicyAgent, QAgent


def selfplay_policy_vs_q(
    board_size: int = 7,
    episodes: int = 500,
    env_params: Optional[dict] = None,
    black_is_policy: bool = True,
    policy_params: Optional[dict] = None,
    q_params: Optional[dict] = None,
) -> Tuple[
    object,
    object,
    List[float],
    List[float],
    List[int],
    List[int],
]:
    """PolicyAgent と QAgent の自己対戦を実行する"""
    if env_params is None:
        env_params = {}
    if policy_params is None:
        policy_params = {}
    if q_params is None:
        q_params = {}

    # --- エージェントと環境の準備 --------------------------------------
    env = GomokuEnv(board_size=board_size, **env_params)

    if black_is_policy:
        black_agent = PolicyAgent(board_size=board_size, **policy_params)
        white_agent = QAgent(board_size=board_size, **q_params)
        desc_str = "Policy(Black) vs Q(White)"
    else:
        black_agent = QAgent(board_size=board_size, **q_params)
        white_agent = PolicyAgent(board_size=board_size, **policy_params)
        desc_str = "Q(Black) vs Policy(White)"

    all_rewards_black: List[float] = []
    all_rewards_white: List[float] = []
    winners: List[int] = []
    turns_list: List[int] = []

    for _ in tqdm(range(episodes), desc=desc_str):
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
                if black_is_policy:
                    black_agent.record_reward(reward)
                    black_episode_reward += reward
                else:
                    black_agent.record_transition(state, action, reward, next_obs, done)
                    black_episode_reward += reward
            else:
                if black_is_policy:
                    white_agent.record_transition(state, action, reward, next_obs, done)
                    white_episode_reward += reward
                else:
                    white_agent.record_reward(reward)
                    white_episode_reward += reward

        if black_is_policy:
            black_agent.finish_episode()
        else:
            pass  # QAgent は特別な後処理なし

        if black_is_policy:
            pass  # 白番は QAgent
        else:
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
