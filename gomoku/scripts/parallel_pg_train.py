# parallel_pg_train.py

"""PolicyAgent用の並列学習機能をまとめたモジュール。

元の parallel_train.py から切り出して可読性を高めた。
"""
import multiprocessing
# CUDA 使用時に fork ベースのマルチプロセスを避けるため
# プール生成時には 'spawn' を指定する
from tqdm import tqdm

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import (
    PolicyAgent,
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent,
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
    """
    ワーカープロセスごとにエージェントを作成し、複数エピソードを実行してデータを返す関数。
      - board_size: 盤サイズ
      - agent_params: PolicyAgent に渡すパラメータ(dict)
      - opponent_class: 対戦相手エージェントのクラス
      - env_params: GomokuEnv のパラメータ(dict)
      - policy_color: PolicyAgent を "black" か "white" のどちらで学習するか
      - initial_state_dict: メインプロセスから渡される初期重み(dict)

    戻り値: local_data (リスト)。要素は (episode_log, winner, turn_count) のタプル。
    """
    if env_params is None:
        env_params = {}

    # 学習対象のPolicyAgent
    policy_agent = PolicyAgent(board_size=board_size, **agent_params)
    # 初期重みが渡されている場合はロード
    if initial_state_dict is not None:
        policy_agent.model.load_state_dict(initial_state_dict)
    # 対戦相手
    opponent_agent = opponent_class()

    # 先手・後手を振り分け
    if policy_color == "black":
        black_agent = policy_agent
        white_agent = opponent_agent
    else:
        black_agent = opponent_agent
        white_agent = policy_agent

    # 環境
    env = GomokuEnv(board_size=board_size, **env_params)

    local_data = []
    for _ in range(num_episodes):
        # 1ゲーム実行
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
    policy_color="black"
):
    """
    メインプロセスが並列でエピソードを集め、バッチ単位でREINFORCE学習する例。

    引数:
      - total_episodes: 全体で実行するエピソード総数
      - batch_size: 1回の学習あたりに収集するエピソード数
      - board_size: 碁盤のサイズ
      - num_workers: 並列ワーカー数 (multiprocessing.Pool)
      - lr, gamma, hidden_size: PolicyAgentの学習ハイパーパラメータ
      - env_params: 環境のパラメータ(dict)。例: {"force_center_first_move":True, "adjacency_range":1, ...}
      - agent_params: 学習対象PolicyAgentの追加パラメータ(dict)。Noneなら内部で作成
      - opponent_class: 対戦相手として使うエージェントのクラス
      - policy_color: "black" なら先手、"white" なら後手を学習させる
        (ワーカーにはこのメインエージェントの重みを渡して同期させる)

    戻り値: (学習済みpolicy_agent, 各エピソードの報酬リスト)
    """
    if env_params is None:
        env_params = {}

    if agent_params is None:
        agent_params = {}
    # デフォルトのパラメータを上書き
    agent_params.setdefault("hidden_size", hidden_size)
    agent_params.setdefault("lr", lr)
    agent_params.setdefault("gamma", gamma)

    # メインエージェント (パラメータの更新先)
    # ここでは学習そのものは最後に "update_with_trajectories" で行うので、
    # このagentインスタンスが「最終的な学習後の重み」を持つことになる。
    policy_agent = PolicyAgent(
        board_size=board_size,
        **agent_params
    )

    # 全エピソードを batch_size ごとに回す
    all_rewards = []
    all_winners = []
    all_turn_counts = []
    n_batches = total_episodes // batch_size

    # CUDA 使用環境では fork 実行だとエラーになるため
    # spawn コンテキストでプールを生成する
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        for _ in tqdm(range(n_batches), desc="Training"):
            # 今バッチで実行するエピソード数 = batch_size
            # これをワーカーに分散
            eppw = batch_size // num_workers

            # 各ワーカーに渡すパラメータ
            # メインエージェントの重み(state_dict)をコピーして送る
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
            # 並列実行 => 各ワーカーが eppwエピソードずつ回して (episode_log, winner) のリストを返す
            results = pool.starmap(train_worker, worker_args)
            # results は [worker_data, worker_data, ...] (長さ num_workers)
            #   worker_data も [ (episode_log, winner), (episode_log, winner), ... ] (長さ eppw)

            # 全ワーカーからの全エピソードをまとめる
            all_episodes = []
            for worker_data in results:
                for (episode_log, winner, turn_count) in worker_data:
                    # 学習対象(PolicyAgent)視点の報酬
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

            # --- バッチ内のエピソードをまとめて学習 ---
            loss = update_with_trajectories(policy_agent, all_episodes)

    return policy_agent, all_rewards, all_winners, all_turn_counts


__all__ = ["play_one_episode", "train_worker", "train_master", "update_with_trajectories"]
