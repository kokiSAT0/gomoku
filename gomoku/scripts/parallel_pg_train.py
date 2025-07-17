# parallel_pg_train.py

"""PolicyAgent用の並列学習機能をまとめたモジュール。

元の parallel_train.py から切り出して可読性を高めた。
"""
import numpy as np
import multiprocessing
# CUDA 使用時に fork ベースのマルチプロセスを避けるため
# プール生成時には 'spawn' を指定する
from tqdm import tqdm
import torch
import torch.nn.functional as F

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import (
    PolicyAgent,
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent,
    EpisodeStep
)

def play_one_episode(env, agent_black, agent_white, policy_color="black"):
    """
    1エピソード(=1ゲーム)を実行し、(episode_log, winner, turn_count) を返す。

    Parameters
    ----------
    env : GomokuEnv
        ゲーム環境
    agent_black : Agent
        先手のエージェント
    agent_white : Agent
        後手のエージェント
    policy_color : str
        "black" なら先手エージェントを学習対象とし、
        "white" なら後手エージェントを学習対象とする。

    学習対象エージェントの行動後に得られた報酬を episode_log に記録して返す。
    """
    obs = env.reset()
    done = False

    # 学習対象エージェントを決定
    policy_agent = agent_black if policy_color == "black" else agent_white

    # 学習対象エージェントのログの開始位置
    start_log_len = len(policy_agent.episode_log)

    while not done:
        if env.current_player == 1:
            action = agent_black.get_action(obs, env)
        else:
            action = agent_white.get_action(obs, env)

        next_obs, reward, done, info = env.step(action)

        # 学習対象の行動後であれば報酬を記録
        if policy_color == "black" and env.current_player == 2:
            policy_agent.record_reward(reward)
        elif policy_color == "white" and env.current_player == 1:
            policy_agent.record_reward(reward)

        obs = next_obs

    winner = info["winner"]
    turn_count = env.turn_count

    # 勝敗が決したあと、学習対象が負けていたら最後の行動に敗北報酬を与える
    if (policy_color == "black" and winner == 2) or (
        policy_color == "white" and winner == 1
    ):
        policy_agent.record_reward(-1.0)
    elif winner == -1:
        policy_agent.record_reward(0.0)

    # 今エピソードで追加された分のログを抽出
    end_log_len = len(policy_agent.episode_log)
    episode_log = policy_agent.episode_log[start_log_len:end_log_len]

    return episode_log, winner, turn_count


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


def update_with_trajectories(agent, all_episodes):
    """
    外部で収集した複数エピソード(all_episodes)をまとめて
    REINFORCEにより学習を行うための処理。

    all_episodes: [episode_log, episode_log, ...]
      episode_log は EpisodeStep のリストまたは
      (state_tensor, action, reward) タプルのリストを想定する。
    agent.device に従いテンソルを GPU / CPU へ転送する。
    """
    import torch
    import torch.nn.functional as F

    all_states = []
    all_actions = []
    all_returns = []

    # 各エピソードごとに割引報酬和を計算
    for episode_log in all_episodes:
        # EpisodeStep オブジェクトかタプルかを判定
        if not episode_log:
            continue

        is_obj = isinstance(episode_log[0], EpisodeStep)

        # --- 割引報酬和の計算 ---
        G = 0.0
        returns = []
        rewards = [step.reward if is_obj else step[2] for step in episode_log]
        for r in reversed(rewards):
            G = agent.gamma * G + r
            returns.insert(0, G)

        # --- 状態・行動・報酬をそれぞれリストへ追加 ---
        for idx, step in enumerate(episode_log):
            s = step.state if is_obj else step[0]
            a = step.action if is_obj else step[1]
            all_states.append(s)
            all_actions.append(a)
            all_returns.append(returns[idx])

    if len(all_states) == 0:
        return 0.0  # 何も学習することがなかった場合

    # --- 各エピソードから集めたデータをテンソルに変換しデバイスへ送る ---
    import torch
    states_tensor = torch.cat(all_states, dim=0).to(agent.device)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long).to(agent.device)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32).to(agent.device)

    logits = agent.model(states_tensor)
    log_probs = F.log_softmax(logits, dim=1)
    chosen_log_probs = log_probs[range(len(actions_tensor)), actions_tensor]

    loss = - (returns_tensor * chosen_log_probs).mean()

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    return loss.item()
