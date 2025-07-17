"""総当たり戦の実行ロジックをまとめたモジュール"""

from typing import Tuple, List, Dict, Any

import numpy as np
from tqdm import tqdm

from ..core.gomoku_env import GomokuEnv


def play_match(agent_black, agent_white, board_size: int = 9, num_episodes: int = 10) -> Tuple[int, int, int]:
    """2つのエージェントを対戦させて結果を集計する"""
    wins = 0
    losses = 0
    draws = 0

    env = GomokuEnv(board_size=board_size)

    for _ in range(num_episodes):
        obs = env.reset()
        done = False

        while not done:
            if env.current_player == 1:  # 黒番
                action = agent_black.get_action(obs, env)
            else:  # 白番
                action = agent_white.get_action(obs, env)

            obs, reward, done, info = env.step(action)

        winner = info["winner"]
        if winner == 1:
            wins += 1
        elif winner == 2:
            losses += 1
        else:
            draws += 1

    return wins, losses, draws


def round_robin_tournament(
    agents: List[Any],
    agent_names: List[str] | None = None,
    board_size: int = 9,
    num_episodes: int = 10,
    show_progress: bool = True,
) -> Tuple[np.ndarray, Dict[tuple, dict], Dict[str, Any]]:
    """複数エージェントの総当たり戦を実行"""
    num_agents = len(agents)
    if agent_names is None:
        agent_names = [f"Agent{i}" for i in range(num_agents)]

    # 勝率を格納する行列。i==j の組み合わせは対戦しないので NaN とする
    result_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
    result_matrix[:] = np.nan

    # 詳細結果を辞書に保存
    detailed_stats: Dict[tuple, dict] = {}

    total_matches = num_agents * (num_agents - 1)
    pbar = tqdm(total=total_matches, desc="RoundRobin", disable=(not show_progress))

    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue

            wins, losses, draws = play_match(
                agent_black=agents[i],
                agent_white=agents[j],
                board_size=board_size,
                num_episodes=num_episodes,
            )
            win_rate = wins / num_episodes

            result_matrix[i][j] = win_rate
            detailed_stats[(i, j)] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate,
            }

            pbar.update(1)
    pbar.close()

    # 黒番としての平均勝率でランキングを作成
    avg_win_rates: List[float] = []
    for i in range(num_agents):
        valid_vals = [x for x in result_matrix[i, :] if not np.isnan(x)]
        mean_wr = float(np.mean(valid_vals)) if valid_vals else 0.0
        avg_win_rates.append(mean_wr)

    ranking = sorted(range(num_agents), key=lambda x: avg_win_rates[x], reverse=True)
    ranking_info = {
        "ranking_order": ranking,
        "avg_win_rates": avg_win_rates,
    }

    return result_matrix, detailed_stats, ranking_info
