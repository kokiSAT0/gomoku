# -*- coding: utf-8 -*-
"""PolicyAgent を LongestChainAgent に対して学習させるスクリプト

必要に応じて複数プロセスを利用することで、学習を高速化できる。
学習過程の報酬や勝率を可視化し、学習後はモデル保存と簡易評価を行う。
"""

from pathlib import Path
import argparse
import torch

from gomoku_env import GomokuEnv
from agents import PolicyAgent, LongestChainAgent
from learning_all_in_one import train_agents, plot_results
from evaluate_models import evaluate_model
from parallel_train import train_master

# モデル保存用ディレクトリ
MODEL_DIR = Path(__file__).resolve().parent / "models"


def main():
    """コマンドライン引数を受け取って学習を実行する"""

    parser = argparse.ArgumentParser(
        description="PolicyAgent を LongestChainAgent と対戦させて学習する"
    )
    parser.add_argument("--episodes", type=int, default=1000, help="学習エピソード数")
    parser.add_argument("--board-size", type=int, default=9, help="盤面サイズ")
    parser.add_argument("--num-workers", type=int, default=1, help="並列ワーカー数")
    parser.add_argument("--device", default=None, help="使用デバイス(cuda/cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    board_size = args.board_size

    if args.num_workers > 1:
        # 並列学習モード
        # train_master() は複数プロセスでエピソードを収集し
        # まとめて方策勾配法で更新を行う
        agent_params = {"device": device}
        black_agent, rewards_b, winners, turns = train_master(
            total_episodes=args.episodes,
            board_size=board_size,
            num_workers=args.num_workers,
            agent_params=agent_params,
            opponent_class=LongestChainAgent,
        )
        # 白番の報酬は黒番の報酬に符号を反転させて近似
        rewards_w = [-r if r != 0 else 0 for r in rewards_b]
    else:
        # 単一プロセスでの学習
        env = GomokuEnv(board_size=board_size)
        black_agent = PolicyAgent(board_size=board_size, device=device)
        white_agent = LongestChainAgent()

        # 単純な1プロセス学習
        print("学習を開始します...")
        rewards_b, rewards_w, winners, turns = train_agents(
            env, black_agent, white_agent, args.episodes
        )

    # モデルを保存
    save_path = MODEL_DIR / "policy_vs_longest.pth"
    black_agent.save_model(save_path)

    # 学習過程の可視化
    plot_results(rewards_b, rewards_w, winners, turns, title="Policy vs LongestChain")

    # 簡易評価
    win_rate = evaluate_model(
        policy_path=save_path,
        opponent_agent=LongestChainAgent(),
        num_episodes=200,
        board_size=board_size,
        device=device,
    )
    print(f"学習後の勝率: {win_rate:.2f}")


if __name__ == "__main__":
    main()

