# parallel_train.py
import numpy as np
import multiprocessing
from tqdm import tqdm
import torch
import torch.nn.functional as F

from gomoku_env import GomokuEnv
from agents import (
    PolicyAgent,
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent
)

def play_one_episode(env, agent_black, agent_white):
    """
    1エピソード(=1ゲーム)を実行し、最終的に (episode_log, winner) を返す。
    
    - 黒番(先手)は agent_black, 白番(後手)は agent_white。
    - 黒番エージェントが方策学習する想定（PolicyAgentなど）。
      黒が打った後に得られた報酬を agent_black.record_reward(reward) で記録していく。
    - ここでは黒番エージェントの内部ログのうち、このゲームで追加された分だけを抽出して返す。
    """
    obs = env.reset()
    done = False

    # 黒番エージェント内部のログの開始位置
    start_log_len = len(agent_black.episode_log)

    while not done:
        if env.current_player == 1:
            action = agent_black.get_action(obs, env)
        else:
            action = agent_white.get_action(obs, env)

        next_obs, reward, done, info = env.step(action)

        # 今手を打ったのが黒なら、その行動の報酬を記録
        if env.current_player == 2:  # 黒番が打った後はcurrent_player==2(白番)になる
            agent_black.record_reward(reward)

        obs = next_obs

    winner = info["winner"]
    turn_count = env.turn_count

    # 今エピソードで追加された分のログを抽出
    end_log_len = len(agent_black.episode_log)
    episode_log = agent_black.episode_log[start_log_len:end_log_len]

    return episode_log, winner, turn_count


def train_worker(
    worker_id,
    num_episodes,
    board_size,
    agent_params,
    opponent_class=ImmediateWinBlockAgent,
    env_params=None
):
    """
    ワーカープロセスごとにエージェントを作成し、複数エピソードを実行してデータを返す関数。
      - board_size: 盤サイズ
      - agent_params: 黒番(PolicyAgentなど)に渡すパラメータ(dict)
      - opponent_class: 対戦相手エージェントのクラス (例: ImmediateWinBlockAgent, RandomAgent等)
      - env_params: GomokuEnvのパラメータ(dict)。force_center_first_move等を指定可能。

    戻り値: local_data (リスト)。要素は (episode_log, winner) のタプル。
    """
    if env_params is None:
        env_params = {}

    # 黒番(学習対象)のPolicyAgentを作る
    policy_agent = PolicyAgent(board_size=board_size, **agent_params)
    # 白番(対戦相手)
    white_agent = opponent_class()

    # 環境
    env = GomokuEnv(board_size=board_size, **env_params)

    local_data = []
    for _ in range(num_episodes):
        # 1ゲーム実行
        episode_log, winner, turn_count = play_one_episode(env, policy_agent, white_agent)
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
    opponent_class=ImmediateWinBlockAgent
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
      - agent_params: 黒番エージェント(PolicyAgent)の追加パラメータ(dict)。Noneなら内部で作成
      - opponent_class: 対戦相手として使うエージェントのクラス

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

    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(range(n_batches), desc="Training"):
            # 今バッチで実行するエピソード数 = batch_size
            # これをワーカーに分散
            eppw = batch_size // num_workers

            # 各ワーカーに渡すパラメータ
            worker_args = []
            for wid in range(num_workers):
                worker_args.append((
                    wid,          # worker_id (未使用)
                    eppw,         # num_episodes (ワーカー単位)
                    board_size,
                    agent_params, # black番のパラメータ
                    opponent_class,
                    env_params
                ))
            # 並列実行 => 各ワーカーが eppwエピソードずつ回して (episode_log, winner) のリストを返す
            results = pool.starmap(train_worker, worker_args)
            # results は [worker_data, worker_data, ...] (長さ num_workers)
            #   worker_data も [ (episode_log, winner), (episode_log, winner), ... ] (長さ eppw)

            # 全ワーカーからの全エピソードをまとめる
            all_episodes = []
            for worker_data in results:
                for (episode_log, winner, turn_count) in worker_data:
                    # black視点の報酬 (+1=黒勝ち, -1=白勝ち, 0=引き分け)
                    if winner == 1:
                        r = 1.0
                    elif winner == 2:
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
      ここで episode_log は [(state_tensor, action, reward), (state_tensor, action, reward), ...]
    """
    import torch
    import torch.nn.functional as F

    all_states = []
    all_actions = []
    all_returns = []

    # 各エピソードごとに割引報酬和を計算
    for episode_log in all_episodes:
        G = 0.0
        returns = []
        for i in reversed(range(len(episode_log))):
            _, _, r = episode_log[i]
            G = agent.gamma * G + r
            returns.insert(0, G)

        for i, (s, a, _) in enumerate(episode_log):
            all_states.append(s)
            all_actions.append(a)
            all_returns.append(returns[i])

    if len(all_states) == 0:
        return 0.0  # 何も学習することがなかった場合

    # テンソル化
    import torch
    states_tensor = torch.cat(all_states, dim=0)       # shape: (N, board_size*board_size)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)  # shape: (N,)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32)  # shape: (N,)

    logits = agent.model(states_tensor)
    log_probs = F.log_softmax(logits, dim=1)
    chosen_log_probs = log_probs[range(len(actions_tensor)), actions_tensor]

    loss = - (returns_tensor * chosen_log_probs).mean()

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    return loss.item()

# 以下はQ学習用
def play_one_episode_q(env, q_agent, opponent_agent):
    """
    QAgent(黒) vs opponent_agent(白) の1ゲーム。
    - env: GomokuEnv
    - q_agent: QAgent
    - opponent_agent: RandomAgent 等
    戻り値: (reward_for_black, winner, turn_count)
    """
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

        # 黒(QAgent)が打った場合のみ、学習データとして保存
        if current_player == 1:
            # QAgentは stepごとに record_transition() して
            # そこでリプレイバッファに溜め、必要に応じて train_on_batch() する
            q_agent.record_transition(
                state, action, reward, next_obs, done
            )

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
    env_params=None
):
    """
    ワーカーがローカルに QAgent を生成し、num_episodes 回の対局を実行して
    各エピソードの (reward_for_black, winner, turn_count) を返す。
    """
    from gomoku_env import GomokuEnv
    if env_params is None:
        env_params = {}

    # 黒(QAgent)
    from agents import QAgent
    q_agent = QAgent(board_size=board_size, **agent_params)
    # 白(対戦相手)
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
    opponent_class=None
):
    """
    QAgent (DQN) を並列で学習する例 (簡易版)。
    注意: QAgentはstepごとに内部学習するため、ワーカー側で学習が進む。
          メインプロセスでパラメータをまとめるなら、あとで平均化など追加要。
    """
    import multiprocessing
    from tqdm import tqdm

    if agent_params is None:
        agent_params = {}
    if env_params is None:
        env_params = {}
    if opponent_class is None:
        from agents import RandomAgent
        opponent_class = RandomAgent

    episodes_per_worker = total_episodes // num_workers

    all_rewards = []
    all_winners = []
    all_turn_counts = []

    # 並列実行
    with multiprocessing.Pool(num_workers) as pool:
        worker_args = []
        for wid in range(num_workers):
            worker_args.append((
                wid,
                episodes_per_worker,
                board_size,
                agent_params,
                opponent_class,
                env_params
            ))
        results = pool.starmap(train_worker_q, worker_args)

    # results は [(local_data, q_agent), (local_data, q_agent), ...] (num_workers要素)
    # local_data: [ (r_for_black, winner, turn_count), ... ]
    # q_agent: 学習済みのQAgent (最後のパラメータを持つ)

    # 各ワーカーの結果をまとめる
    #   (本格的にパラメータを同期したいならさらに工夫が必要)
    for (local_data, worker_q_agent) in results:
        for (r, w, t) in local_data:
            all_rewards.append(r)
            all_winners.append(w)
            all_turn_counts.append(t)

    # とりあえず先頭のワーカーで学習したエージェントを返す例
    final_agent = results[0][1]

    return final_agent, all_rewards, all_winners, all_turn_counts
