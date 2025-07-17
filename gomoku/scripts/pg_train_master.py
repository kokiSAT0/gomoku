"""PolicyAgent の並列学習用メイン関数をまとめたモジュール

train_worker と train_master を定義する。parallel_pg_train.py から
切り出すことで構成を分かりやすくした。
"""

import multiprocessing
from tqdm import tqdm

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import (
    PolicyAgent,
    ImmediateWinBlockAgent,
)

from .pg_train_utils import play_one_episode, update_with_trajectories


def train_worker(
    worker_id,
    num_episodes,
    board_size,
    agent_params,
    opponent_class=ImmediateWinBlockAgent,
    env_params=None,
    policy_color="black",
    initial_state_dict=None,
):
    """ワーカー側でエピソードを実行しデータを返す関数"""
    if env_params is None:
        env_params = {}

    # 学習対象エージェントを生成
    policy_agent = PolicyAgent(board_size=board_size, **agent_params)

    # メインプロセスから渡された重みがあれば読み込む
    if initial_state_dict is not None:
        policy_agent.model.load_state_dict(initial_state_dict)

    opponent_agent = opponent_class()

    # 先手・後手を決定
    if policy_color == "black":
        black_agent = policy_agent
        white_agent = opponent_agent
    else:
        black_agent = opponent_agent
        white_agent = policy_agent

    env = GomokuEnv(board_size=board_size, **env_params)

    local_data = []
    for _ in range(num_episodes):
        # 1ゲーム実行しログを収集
        episode_log, winner, turn_count = play_one_episode(
            env,
            black_agent,
            white_agent,
            policy_color=policy_color,
        )
        local_data.append((episode_log, winner, turn_count))

    return local_data


def train_master(
    total_episodes=1000,
    batch_size=50,
    board_size=9,
    num_workers=4,
    lr=1e-3,
    gamma=0.95,
    hidden_size=128,
    env_params=None,
    agent_params=None,
    opponent_class=ImmediateWinBlockAgent,
    policy_color="black",
):
    """複数ワーカーでデータを集め PolicyAgent を学習する簡易例"""
    if env_params is None:
        env_params = {}

    if agent_params is None:
        agent_params = {}

    # デフォルトパラメータを設定
    agent_params.setdefault("hidden_size", hidden_size)
    agent_params.setdefault("lr", lr)
    agent_params.setdefault("gamma", gamma)

    # 学習対象エージェント (パラメータの更新先)
    policy_agent = PolicyAgent(board_size=board_size, **agent_params)

    all_rewards = []
    all_winners = []
    all_turn_counts = []
    n_batches = total_episodes // batch_size

    # CUDA 環境でも安全な spawn 方式でプールを作成
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        for _ in tqdm(range(n_batches), desc="Training"):
            # 1バッチ内で各ワーカーが担当するエピソード数
            eppw = batch_size // num_workers

            worker_args = []
            for wid in range(num_workers):
                worker_args.append(
                    (
                        wid,
                        eppw,
                        board_size,
                        agent_params,
                        opponent_class,
                        env_params,
                        policy_color,
                        {k: v.cpu() for k, v in policy_agent.model.state_dict().items()},
                    )
                )

            # 各ワーカーでエピソードを収集
            results = pool.starmap(train_worker, worker_args)

            # 収集データをまとめて学習用リストへ
            all_episodes = []
            for worker_data in results:
                for (episode_log, winner, turn_count) in worker_data:
                    if policy_color == "black":
                        if winner == 1:
                            r = 1.0
                        elif winner == 2:
                            r = -1.0
                        else:
                            r = 0.0
                    else:
                        if winner == 2:
                            r = 1.0
                        elif winner == 1:
                            r = -1.0
                        else:
                            r = 0.0

                    all_rewards.append(r)
                    all_winners.append(winner)
                    all_turn_counts.append(turn_count)

                    all_episodes.append(episode_log)

            # バッチ分のログで学習
            loss = update_with_trajectories(policy_agent, all_episodes)

    return policy_agent, all_rewards, all_winners, all_turn_counts


__all__ = ["train_worker", "train_master"]
