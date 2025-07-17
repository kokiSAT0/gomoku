"""総当たり戦の結果可視化関数をまとめたモジュール"""

from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.utils import FIGURE_DIR


def print_tournament_results(
    result_matrix: np.ndarray,
    detailed_stats: Dict[tuple, dict],
    ranking_info: Dict[str, Any],
    agent_names: List[str],
) -> None:
    """総当たり戦の成績をテキストで表示"""
    num_agents = len(agent_names)
    print("\n====== Round Robin Result (Black vs White) ======")
    print("Rows = Black, Cols = White (Value=BlackWinRate)")
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

    ranking = ranking_info["ranking_order"]
    avg_win_rates = ranking_info["avg_win_rates"]

    print("\n====== Ranking by Black-WinRate ======")
    for rank, idx in enumerate(ranking, start=1):
        print(f"Rank {rank}: {agent_names[idx]} with average black-win-rate = {avg_win_rates[idx]:.3f}")



def plot_winrate_heatmap(result_matrix: np.ndarray, agent_names: List[str], show: bool = True) -> Path:
    """勝率行列をヒートマップとして保存・表示"""
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.isnan(result_matrix)
    sns.heatmap(
        result_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        mask=mask,
        xticklabels=agent_names,
        yticklabels=agent_names,
        cbar=True,
        square=True,
        ax=ax,
    )
    ax.set_xlabel("White")
    ax.set_ylabel("Black")
    ax.set_title("Black Win Rate Matrix")
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"winrate_heatmap_{timestamp}.png"
    path = FIGURE_DIR / filename
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_ranking_bar(ranking_info: Dict[str, Any], agent_names: List[str], show: bool = True) -> Path:
    """ランキング情報を棒グラフで保存・表示"""
    ranking = ranking_info["ranking_order"]
    avg_win_rates = ranking_info["avg_win_rates"]

    sorted_names = [agent_names[i] for i in ranking]
    sorted_scores = [avg_win_rates[i] for i in ranking]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(ranking)), sorted_scores[::-1], color="skyblue")
    plt.yticks(range(len(ranking)), sorted_names[::-1])
    plt.xlabel("Average Black Win Rate")
    plt.title("Ranking by Black Perspective")
    plt.xlim(0, 1)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ranking_bar_{timestamp}.png"
    path = FIGURE_DIR / filename
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    return path
