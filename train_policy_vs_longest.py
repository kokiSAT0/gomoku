# -*- coding: utf-8 -*-
"""PolicyAgentをLongestChainAgentに対して学習させるスクリプト

学習過程の報酬や勝率を可視化する。学習終了後にはモデルを保存し、
簡易的な評価も行う。
"""

from pathlib import Path
from gomoku_env import GomokuEnv
from agents import PolicyAgent, LongestChainAgent
from learning_all_in_one import train_agents, plot_results
from evaluate_models import evaluate_model

# モデル保存用ディレクトリ
MODEL_DIR = Path(__file__).resolve().parent / "models"


def main():
    """学習実行用メイン関数"""
    # ----- ハイパーパラメータ設定 -----
    board_size = 9
    episodes = 1000  # 必要に応じて増減させる

    # 環境生成 (特に追加ルールは設けない)
    env = GomokuEnv(board_size=board_size)

    # 黒番に学習するPolicyAgent、白番にLongestChainAgentを配置
    black_agent = PolicyAgent(board_size=board_size)
    white_agent = LongestChainAgent()

    # ----- 学習開始 -----
    print("学習を開始します...")
    rewards_b, rewards_w, winners, turns = train_agents(
        env, black_agent, white_agent, episodes
    )

    # モデルを保存
    save_path = MODEL_DIR / "policy_vs_longest.pth"
    black_agent.save_model(save_path)

    # ----- 学習過程の可視化 -----
    plot_results(rewards_b, rewards_w, winners, turns,
                 title="Policy vs LongestChain")

    # ----- 簡易評価 -----
    win_rate = evaluate_model(
        policy_path=save_path,
        opponent_agent=LongestChainAgent(),
        num_episodes=200,
        board_size=board_size,
    )
    print(f"学習後の勝率: {win_rate:.2f}")


if __name__ == "__main__":
    main()
