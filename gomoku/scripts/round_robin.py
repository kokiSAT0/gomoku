"""複数エージェントの総当たり戦を実行し結果を表示するスクリプト"""

from pathlib import Path

from .tournament_core import round_robin_tournament
from .tournament_plot import (
    print_tournament_results,
    plot_winrate_heatmap,
    plot_ranking_bar,
)

# 学習済みモデルの保存先ディレクトリ
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


if __name__ == "__main__":
    from ..ai.agents import (
        RandomAgent,
        PolicyAgent,
        QAgent,
        ImmediateWinBlockAgent,
        FourThreePriorityAgent,
        LongestChainAgent,
    )

    agentA = RandomAgent()
    agentB = ImmediateWinBlockAgent()
    agentC = FourThreePriorityAgent()
    agentD = LongestChainAgent()

    agentP = PolicyAgent(board_size=9, network_type="dense")
    agentP.load_model(MODEL_DIR / "policy_agent_black.pth")

    agentQ = QAgent(board_size=9)
    agentQ.load_model(MODEL_DIR / "q_agent_black.pth")

    agents = [agentA, agentB, agentC, agentD, agentP, agentQ]
    agent_names = ["Random", "Immediate", "FourThree", "Longest", "Policy", "QAgent"]

    result_mat, stats, ranking_info = round_robin_tournament(
        agents=agents,
        agent_names=agent_names,
        board_size=9,
        num_episodes=10,
        show_progress=True,
    )

    print_tournament_results(result_mat, stats, ranking_info, agent_names)
    plot_winrate_heatmap(result_mat, agent_names, show=False)
    plot_ranking_bar(ranking_info, agent_names, show=False)
