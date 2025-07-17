# -*- coding: utf-8 -*-
"""PolicyAgent を LongestChainAgent に対して学習させるスクリプト

必要に応じて複数プロセスを利用することで、学習を高速化できる。
学習過程の報酬や勝率を可視化し、学習後はモデル保存と簡易評価を行う。

本スクリプトでは ``plot_results()`` を ``show=False`` で呼び出し、
GUI の無い環境でも学習グラフを ``figures/`` フォルダへ保存できるようにしている。
"""

from pathlib import Path
import argparse
import torch
import multiprocessing
import sys

# ------------------------------------------------------------
# スクリプトが ``python train_policy_vs_longest.py`` のように
# パッケージ外から直接実行された場合、相対インポートが失敗する。
# そのため ``sys.path`` にプロジェクトルートを追加し、
# ``__package__`` を設定することで相対インポートを可能にする。
# ------------------------------------------------------------
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    __package__ = "gomoku.scripts"

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import PolicyAgent, LongestChainAgent
from .learning_utils import train_agents, plot_results
from .evaluate_models import evaluate_model
from .parallel_train import train_master

# モデル保存用ディレクトリ
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# -----------------------------------------------
# 3連や4連をブロックした時に高めの報酬を与える
# 環境設定ディクショナリ
# -----------------------------------------------
ENV_PARAMS = {
    # 両端が空いている四連をブロックした場合の報酬
    "reward_block_4_open2": 1.5,
    # 片端のみ空いている四連をブロックした場合の報酬
    "reward_block_4_open1": 1.0,
    # 両端が空いている三連をブロックした場合の報酬
    "reward_block_3_open2": 0.8,
    # 片端のみ空いている三連をブロックした場合の報酬
    "reward_block_3_open1": 0.5,
}


def main():
    """コマンドライン引数を受け取って学習を実行する"""

    parser = argparse.ArgumentParser(
        description="PolicyAgent を LongestChainAgent と対戦させて学習する"
    )
    parser.add_argument("--episodes", type=int, default=1000, help="学習エピソード数")
    parser.add_argument("--board-size", type=int, default=9, help="盤面サイズ")
    parser.add_argument("--num-workers", type=int, default=10, help="並列ワーカー数")
    parser.add_argument("--device", default=None, help="使用デバイス(cuda/cpu)")
    parser.add_argument(
        "--policy-color",
        choices=["black", "white"],
        default="black",
        help="PolicyAgent を先手(black)か後手(white)のどちらで学習するか",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    board_size = args.board_size
    policy_color = args.policy_color

    # CUDA 使用時に fork 方式で子プロセスを起動するとエラーになるため
    # 並列実行を行う場合は start_method を 'spawn' に設定する
    if args.num_workers > 1:
        multiprocessing.set_start_method("spawn", force=True)

    if args.num_workers > 1:
        # 並列学習モード
        # train_master() は複数プロセスでエピソードを収集し
        # まとめて方策勾配法で更新を行う
        agent_params = {"device": device}
        policy_agent, rewards_p, winners, turns = train_master(
            total_episodes=args.episodes,
            board_size=board_size,
            num_workers=args.num_workers,
            agent_params=agent_params,
            opponent_class=LongestChainAgent,
            policy_color=policy_color,
            env_params=ENV_PARAMS,
        )
        if policy_color == "black":
            rewards_b = rewards_p
            rewards_w = [-r if r != 0 else 0 for r in rewards_p]
        else:
            rewards_w = rewards_p
            rewards_b = [-r if r != 0 else 0 for r in rewards_p]
    else:
        # 単一プロセスでの学習
        # ブロック報酬を組み込んだ環境を生成
        env = GomokuEnv(board_size=board_size, **ENV_PARAMS)
        if policy_color == "black":
            black_agent = PolicyAgent(board_size=board_size, device=device)
            white_agent = LongestChainAgent()
            policy_agent = black_agent
        else:
            black_agent = LongestChainAgent()
            white_agent = PolicyAgent(board_size=board_size, device=device)
            policy_agent = white_agent

        # 単純な1プロセス学習
        print("学習を開始します...")
        rewards_b, rewards_w, winners, turns = train_agents(
            env, black_agent, white_agent, args.episodes
        )

    # モデルを保存 (学習対象エージェント)
    save_path = MODEL_DIR / "policy_vs_longest.pth"
    policy_agent.save_model(save_path)

    # ------------------------------------------------------------
    # 学習過程の可視化
    #   GUI の無い環境でも画像ファイルとして保存できるよう
    #   show=False を指定して表示を省略する
    # ------------------------------------------------------------
    plot_results(
        rewards_b,
        rewards_w,
        winners,
        turns,
        title="Policy vs LongestChain",
        show=False,
    )

    # 簡易評価
    win_rate = evaluate_model(
        policy_path=save_path,
        opponent_agent=LongestChainAgent(),
        num_episodes=200,
        board_size=board_size,
        device=device,
        policy_color=policy_color,
    )
    print(f"学習後の勝率: {win_rate:.2f}")


if __name__ == "__main__":
    main()

