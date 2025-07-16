# round_robin.py

"""複数のエージェントを総当たり戦させて勝率表を作成するスクリプト。

`round_robin_tournament` 関数を利用して全ての組み合わせを評価し、
成績やランキングを算出する。
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from gomoku_env import GomokuEnv

# 学習済みモデルの保存先ディレクトリ
MODEL_DIR = Path(__file__).resolve().parent / "models"

def play_match(agent_black, agent_white, board_size=9, num_episodes=10):
    """
    agent_black(黒番) と agent_white(白番) を対戦させ、
    指定した回数(num_episodes)だけゲームを行い、
    黒番視点での (wins, losses, draws) を返す。
      - wins: 黒番勝ち回数
      - losses: 黒番負け(=白番勝ち)回数
      - draws: 引き分け回数
    """
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
            else:                       # 白番
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
    agents,
    agent_names=None,
    board_size=9,
    num_episodes=10,
    show_progress=True
):
    """
    agents: [agent0, agent1, agent2, ...] のリスト
      - 各 agent は get_action(obs, env) メソッドを持つ想定 (PolicyAgent, QAgent, ヒューリスティックなど)
    agent_names: ["name0", "name1", "name2", ...] のリスト (任意)
      - 指定がなければ 'Agent0', 'Agent1', ... と名付ける
    board_size: 盤面サイズ
    num_episodes: 各対戦(黒番 vs 白番)ごとの試合数
    show_progress: Trueなら tqdm で進捗表示

    戻り値:
      - result_matrix: shape=(N, N) の2次元配列 (黒番視点の勝率表)
                      result_matrix[i][j]: i番エージェントが黒番, j番エージェントが白番のときの 黒番勝率(0~1)
      - detailed_stats: {(i,j): {"wins":..., "losses":..., "draws":..., "win_rate":...}, ...}
      - ranking_info: エージェントごとの総合評価 (後述)
    """
    num_agents = len(agents)
    if agent_names is None:
        agent_names = [f"Agent{i}" for i in range(num_agents)]

    # 結果を格納する行列
    # result_matrix[i][j] = "i番エージェントが黒番 / j番エージェントが白番" での黒番勝率
    result_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
    result_matrix[:] = np.nan  # i==jの場合は対戦なしなのでNaNにする

    # 詳細な辞書: {(i,j): {"wins":..., "losses":..., "draws":..., "win_rate":...}, ...}
    detailed_stats = {}

    # 対戦総数 (i!=j の組み合わせ)
    total_matches = num_agents * (num_agents - 1)

    # tqdm で進捗を表示
    # "desc" でラベル、 total=総対戦カード数
    pbar = tqdm(total=total_matches, desc="RoundRobin", disable=(not show_progress))

    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue

            # i番(黒) vs j番(白)
            wins, losses, draws = play_match(
                agent_black=agents[i],
                agent_white=agents[j],
                board_size=board_size,
                num_episodes=num_episodes
            )
            win_rate = wins / num_episodes

            result_matrix[i][j] = win_rate
            detailed_stats[(i, j)] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate
            }

            pbar.update(1)  # 1試合カード終了
    pbar.close()

    # --- 各エージェントのランキングを計算 ---
    # 今回は「黒番としての平均勝率」で評価 (row平均)
    avg_win_rates = []
    for i in range(num_agents):
        # row i (iが黒番)の有効値(=相手がjでj!=i)を取り出し平均
        valid_vals = [x for x in result_matrix[i, :] if not np.isnan(x)]
        if len(valid_vals) == 0:
            mean_wr = 0.0
        else:
            mean_wr = float(np.mean(valid_vals))
        avg_win_rates.append(mean_wr)

    # ランキング: 黒番平均勝率が高い順
    ranking = sorted(range(num_agents), key=lambda x: avg_win_rates[x], reverse=True)
    ranking_info = {
        "ranking_order": ranking,  # インデックスのリスト
        "avg_win_rates": avg_win_rates
    }

    return result_matrix, detailed_stats, ranking_info


def print_tournament_results(
    result_matrix,
    detailed_stats,
    ranking_info,
    agent_names
):
    """
    総当たり戦の結果をテキストでわかりやすく表示
      - 勝率行列
      - 各カードの詳細 (省略可)
      - ランキング一覧
    """
    num_agents = len(agent_names)
    print("\n====== Round Robin Result (Black vs White) ======")
    print("Rows = Black, Cols = White (Value=BlackWinRate)")
    # 行列を少し整形して出力
    header = ["      "] + [f"{name:15s}" for name in agent_names]
    print("".join(header))
    for i in range(num_agents):
        row_str = f"{agent_names[i]:6s}"
        for j in range(num_agents):
            if np.isnan(result_matrix[i, j]):
                row_str += "      -         "
            else:
                row_str += f"  {result_matrix[i, j]:8.2f}   "
        print(row_str)

    # ランキング情報
    ranking = ranking_info["ranking_order"]
    avg_win_rates = ranking_info["avg_win_rates"]

    print("\n====== Ranking by Black-WinRate ======")
    for rank, idx in enumerate(ranking, start=1):
        print(f"Rank {rank}: {agent_names[idx]} with average black-win-rate = {avg_win_rates[idx]:.3f}")

    # detailed_stats は必要に応じて個別に表示する
    # すべて出すと長いので省略してもOK
    # 例として少しだけフォーマット
    # for (i,j), info in detailed_stats.items():
    #     print(f"{agent_names[i]}(Black) vs {agent_names[j]}(White): {info}")


def plot_winrate_heatmap(result_matrix, agent_names):
    """
    黒番勝率のヒートマップを matplotlib で可視化。
    行(黒番), 列(白番)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8,6))
    # マスク対策: i==j で NaN になっている箇所は表示しない
    mask = np.isnan(result_matrix)

    sns.heatmap(
        result_matrix,
        annot=True,  # セル内に数値表示
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
        mask=mask,
        xticklabels=agent_names,
        yticklabels=agent_names,
        cbar=True,
        square=True,
        ax=ax
    )
    ax.set_xlabel("White")
    ax.set_ylabel("Black")
    ax.set_title("Black Win Rate Matrix")
    plt.tight_layout()
    plt.show()


def plot_ranking_bar(ranking_info, agent_names):
    """
    ランキング(黒番勝率の平均)を棒グラフで可視化。
    """
    ranking = ranking_info["ranking_order"]
    avg_win_rates = ranking_info["avg_win_rates"]

    sorted_names = [agent_names[i] for i in ranking]
    sorted_scores = [avg_win_rates[i] for i in ranking]

    plt.figure(figsize=(8,6))
    plt.barh(range(len(ranking)), sorted_scores[::-1], color="skyblue")
    # ラベル: 下から上にかけて
    plt.yticks(range(len(ranking)), sorted_names[::-1])
    plt.xlabel("Average Black Win Rate")
    plt.title("Ranking by Black Perspective")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 実行例
# ------------------------------------------------------------------
if __name__ == "__main__":
    from agents import (
        RandomAgent,
        PolicyAgent,
        QAgent,
        ImmediateWinBlockAgent,
        FourThreePriorityAgent,
        LongestChainAgent
    )

    # エージェントのリスト
    agentA = RandomAgent()
    agentB = ImmediateWinBlockAgent()
    agentC = FourThreePriorityAgent()
    agentD = LongestChainAgent()

    # 例: 学習済みPolicyAgent (読み込めるならアンコメント)
    agentP = PolicyAgent(board_size=9)
    agentP.load_model(MODEL_DIR / "policy_agent_black.pth")

    # 例: 学習済みQAgent (読み込めるならアンコメント)
    agentQ = QAgent(board_size=9)
    agentQ.load_model(MODEL_DIR / "q_agent_black.pth")

    agents = [agentA, agentB, agentC, agentD, agentP, agentQ]
    agent_names = ["Random", "Immediate", "FourThree", "Longest", "Policy", "QAgent"]

    # 総当たり試合を実行(各マッチ10試合ずつ)
    result_mat, stats, ranking_info = round_robin_tournament(
        agents=agents,
        agent_names=agent_names,
        board_size=9,
        num_episodes=10,
        show_progress=True
    )

    # 結果を表示
    print_tournament_results(result_mat, stats, ranking_info, agent_names)

    # ヒートマップ表示
    plot_winrate_heatmap(result_mat, agent_names)

    # バーグラフ(ランキング)表示
    plot_ranking_bar(ranking_info, agent_names)
