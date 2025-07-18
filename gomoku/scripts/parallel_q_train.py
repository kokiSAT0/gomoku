# parallel_q_train.py

"""QAgent用の並列学習機能をまとめたモジュール。

元の parallel_train.py から切り出して可読性を高めた。
"""

import multiprocessing
from tqdm import tqdm

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import QAgent, RandomAgent


def _update_epsilon(agent: QAgent, steps: int) -> None:
    """epsilon を自前で更新する補助関数"""
    for _ in range(steps):
        agent.epsilon_step += 1
        ratio = min(1.0, agent.epsilon_step / agent.epsilon_decay)
        agent.epsilon = (1.0 - ratio) * agent.epsilon + ratio * agent.epsilon_end


def play_one_episode_q(env, q_agent, opponent_agent, collect_transitions=False):
    """QAgent(黒)と opponent_agent(白) の1ゲームを実行する。

    ``collect_transitions`` が True のときは学習を行わず、遷移情報を
    返すだけの動作になる。"""
    obs = env.reset()
    done = False
    transitions = []

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

        # 黒(QAgent)が打った場合の処理
        if current_player == 1:
            if collect_transitions:
                transitions.append((state, action, reward, next_obs, done))
            else:
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

    if collect_transitions:
        return r_for_black, winner, turn_count, transitions
    return r_for_black, winner, turn_count


def train_worker_q(
    worker_id,
    num_episodes,
    board_size,
    agent_params,
    model_state,
    opponent_class,
    env_params=None,
):
    """ワーカー側で QAgent を生成し複数エピソードを実行する。

    ``model_state`` で渡された重みをロードし、学習は行わずに遷移
    情報のみを収集する。"""
    if env_params is None:
        env_params = {}

    q_agent = QAgent(board_size=board_size, **agent_params)
    q_agent.qnet.load_state_dict(model_state)
    white_agent = opponent_class()

    env = GomokuEnv(board_size=board_size, **env_params)

    local_data = []
    transitions = []
    for _ in range(num_episodes):
        r_for_black, winner, turn_count, trans = play_one_episode_q(
            env,
            q_agent,
            white_agent,
            collect_transitions=True,
        )
        local_data.append((r_for_black, winner, turn_count))
        transitions.extend(trans)
    return local_data, transitions


def train_master_q(
    total_episodes=1000,
    batch_size=50,
    board_size=9,
    num_workers=4,
    agent_params=None,
    env_params=None,
    opponent_class=None,
):
    """QAgent を並列で学習する簡易例。

    1 つの QAgent をマスター側で保持し、各ワーカーはモデルの重みを
    受け取って遷移のみを収集する。集めた遷移はマスターでまとめて
    ``train_on_batch`` を呼び出しながら学習を進める。"""
    if agent_params is None:
        agent_params = {}
    if env_params is None:
        env_params = {}
    if opponent_class is None:
        opponent_class = RandomAgent

    episodes_per_worker = total_episodes // num_workers

    # マスターで共有する QAgent を生成
    q_agent = QAgent(board_size=board_size, **agent_params)

    all_rewards = []
    all_winners = []
    all_turn_counts = []

    # 並列実行。CUDA 利用時もエラーにならないよう spawn 方式を選ぶ
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        worker_args = []
        # モデルの重みは CPU テンソルへ変換して渡す
        model_state = {k: v.detach().cpu() for k, v in q_agent.qnet.state_dict().items()}
        for wid in range(num_workers):
            worker_args.append(
                (
                    wid,
                    episodes_per_worker,
                    board_size,
                    agent_params,
                    model_state,
                    opponent_class,
                    env_params,
                )
            )
        results = pool.starmap(train_worker_q, worker_args)

    # 取得した遷移をマスターのエージェントへ集約
    for (local_data, transitions) in results:
        for (r, w, t) in local_data:
            all_rewards.append(r)
            all_winners.append(w)
            all_turn_counts.append(t)
        for trans in transitions:
            q_agent.buffer.push(*trans)
            _update_epsilon(q_agent, 1)
            if len(q_agent.buffer) >= batch_size:
                q_agent.train_on_batch()

    return q_agent, all_rewards, all_winners, all_turn_counts
