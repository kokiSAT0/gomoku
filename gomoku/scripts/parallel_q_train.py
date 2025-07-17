# parallel_q_train.py

"""QAgent用の並列学習機能をまとめたモジュール。

元の parallel_train.py から切り出して可読性を高めた。
"""

import multiprocessing
from tqdm import tqdm

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import QAgent, RandomAgent


def play_one_episode_q(env, q_agent, opponent_agent):
    """QAgent(黒)と opponent_agent(白) の1ゲームを実行する。"""
    obs = env.reset()
    done = False

    while not done:
        current_player = env.current_player
        state = obs.copy()

        # 行動選択
        if current_player == 1:
            action = q_agent.get_action(obs, env)
        else:
            action = opponent_agent.get_action(obs, env)

        # 1手進める
        next_obs, reward, done, info = env.step(action)

        # 黒(QAgent)が打った場合のみ学習データとして保存
        if current_player == 1:
            q_agent.record_transition(state, action, reward, next_obs, done)

        obs = next_obs

    winner = info["winner"]
    turn_count = env.turn_count

    # 黒視点の最終報酬
    if winner == 1:
        r_for_black = 1.0
    elif winner == 2:
        r_for_black = -1.0
    else:
        r_for_black = 0.0

    return r_for_black, winner, turn_count


def train_worker_q(
    worker_id,
    num_episodes,
    board_size,
    agent_params,
    opponent_class,
    env_params=None,
):
    """ワーカー側で QAgent を生成し複数エピソードを実行する。"""
    if env_params is None:
        env_params = {}

    q_agent = QAgent(board_size=board_size, **agent_params)
    white_agent = opponent_class()

    env = GomokuEnv(board_size=board_size, **env_params)

    local_data = []
    for _ in range(num_episodes):
        r_for_black, winner, turn_count = play_one_episode_q(env, q_agent, white_agent)
        local_data.append((r_for_black, winner, turn_count))
    return local_data, q_agent


def train_master_q(
    total_episodes=1000,
    batch_size=50,
    board_size=9,
    num_workers=4,
    agent_params=None,
    env_params=None,
    opponent_class=None,
):
    """QAgent を並列で学習する簡易例。"""
    if agent_params is None:
        agent_params = {}
    if env_params is None:
        env_params = {}
    if opponent_class is None:
        opponent_class = RandomAgent

    episodes_per_worker = total_episodes // num_workers

    all_rewards = []
    all_winners = []
    all_turn_counts = []

    # 並列実行。CUDA 利用時もエラーにならないよう spawn 方式を選ぶ
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        worker_args = []
        for wid in range(num_workers):
            worker_args.append(
                (
                    wid,
                    episodes_per_worker,
                    board_size,
                    agent_params,
                    opponent_class,
                    env_params,
                )
            )
        results = pool.starmap(train_worker_q, worker_args)

    # 各ワーカーの結果をまとめる
    for (local_data, worker_q_agent) in results:
        for (r, w, t) in local_data:
            all_rewards.append(r)
            all_winners.append(w)
            all_turn_counts.append(t)

    # とりあえず先頭のワーカーで学習したエージェントを返す
    final_agent = results[0][1]

    return final_agent, all_rewards, all_winners, all_turn_counts
