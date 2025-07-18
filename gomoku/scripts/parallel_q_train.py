# parallel_q_train.py

"""QAgent用の並列学習機能をまとめたモジュール。

元の parallel_train.py から切り出して可読性を高めた。
"""

import multiprocessing
from math import ceil
from tqdm import tqdm

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import QAgent, RandomAgent


def _update_epsilon(agent: QAgent, steps: int) -> None:
    """epsilon を自前で更新する補助関数"""
    for _ in range(steps):
        agent.epsilon_step += 1
        ratio = min(1.0, agent.epsilon_step / agent.epsilon_decay)
        agent.epsilon = (1.0 - ratio) * agent.epsilon + ratio * agent.epsilon_end


def _split_episodes(total: int, num_workers: int) -> list[int]:
    """エピソード数をワーカー数で均等に分割する補助関数"""
    base = total // num_workers
    rem = total % num_workers
    return [base + 1 if i < rem else base for i in range(num_workers)]


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
    show_progress: bool = False,
    q_agent: QAgent | None = None,
    pool=None,
):
    """QAgent を並列で学習する簡易例。

    q_agent を引数で受け取った場合はそのまま利用し、新しく生成しない。
    学習を継続したいときに便利。

    1 つの QAgent をマスター側で保持し、各ワーカーはモデルの重みを
    受け取って遷移のみを収集する。集めた遷移はマスターでまとめて
    ``train_on_batch`` を呼び出しながら学習を進める。

    Parameters
    ----------
    show_progress : bool
        tqdm による進捗バーを表示するかどうか
    pool : multiprocessing.pool.Pool | None
        既に生成済みのプールを与えるとそのプールを利用する。``None`` の
        場合はこの関数内で新しくプールを作成し、処理終了後にクローズ
        する。
    """
    if agent_params is None:
        agent_params = {}
    if env_params is None:
        env_params = {}
    if opponent_class is None:
        opponent_class = RandomAgent

    # q_agent が渡されていない場合のみ新規に生成する
    # 既存エージェントを使うことで学習状態を引き継げる
    if q_agent is None:
        q_agent = QAgent(board_size=board_size, **agent_params)

    all_rewards = []
    all_winners = []
    all_turn_counts = []

    n_batches = ceil(total_episodes / batch_size)
    remaining = total_episodes

    def _run(active_pool):
        """与えられたプールで実際の並列処理を行う内部関数"""
        pbar = tqdm(total=total_episodes, desc="Training", disable=(not show_progress))
        for _ in range(n_batches):
            batch_eps = min(batch_size, remaining)
            remaining -= batch_eps

            # 現在のモデル重みを取得し CPU テンソルとして渡す
            model_state = {k: v.detach().cpu() for k, v in q_agent.qnet.state_dict().items()}

            # エピソード数をワーカーへ均等に割り当てる
            per_worker = _split_episodes(batch_eps, num_workers)
            worker_args = []
            for wid, n_ep in enumerate(per_worker):
                if n_ep == 0:
                    continue
                worker_args.append(
                    (
                        wid,
                        n_ep,
                        board_size,
                        agent_params,
                        model_state,
                        opponent_class,
                        env_params,
                    )
                )

            results = active_pool.starmap(train_worker_q, worker_args)

            # 集約した遷移で学習
            for (local_data, transitions) in results:
                for (r, w, t) in local_data:
                    all_rewards.append(r)
                    all_winners.append(w)
                    all_turn_counts.append(t)
                for trans in transitions:
                    q_agent.buffer.push(*trans)
                    _update_epsilon(q_agent, 1)
                    # QAgent 自身が持つバッチサイズに達しているかを確認してから学習
                    # (train_master_q の batch_size 引数とは異なる点に注意)
                    if len(q_agent.buffer) >= q_agent.batch_size:
                        q_agent.train_on_batch()

            pbar.update(batch_eps)
        pbar.close()

    # プールが渡されていない場合だけ自前で生成し終了時に閉じる
    if pool is None:
        with multiprocessing.get_context("spawn").Pool(num_workers) as p:
            _run(p)
    else:
        _run(pool)

    # 更新された q_agent をそのまま返す
    return q_agent, all_rewards, all_winners, all_turn_counts
