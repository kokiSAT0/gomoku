# -*- coding: utf-8 -*-
"""複数の相手と対戦しながら PolicyAgent を学習するスクリプト

ヒューリスティックエージェント(LongestChainAgent)と
事前学習済み PolicyAgent をランダムに選んで対戦させ、
それぞれに勝てるエージェントを目指す。
"""

import argparse
import random
from pathlib import Path
import torch

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import PolicyAgent, LongestChainAgent
from .evaluate_models import evaluate_model

# モデル保存先ディレクトリ
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def run_episode(env, policy_agent, opponent, policy_color="black"):
    """1 エピソードだけ実行して学習用情報を記録する"""
    obs = env.reset()
    done = False

    # 先手・後手の割り当て
    if policy_color == "black":
        black_agent = policy_agent
        white_agent = opponent
    else:
        black_agent = opponent
        white_agent = policy_agent

    while not done:
        current_player = env.current_player
        if current_player == 1:
            action = black_agent.get_action(obs, env)
        else:
            action = white_agent.get_action(obs, env)

        next_obs, reward, done, info = env.step(action)

        # PolicyAgent が実際に打った手に対する報酬を記録
        if (policy_color == "black" and current_player == 1) or (
            policy_color == "white" and current_player == 2
        ):
            policy_agent.record_reward(reward)

        obs = next_obs

    winner = info["winner"]
    # 敗北時には最後の行動にペナルティ
    if (policy_color == "black" and winner == 2) or (
        policy_color == "white" and winner == 1
    ):
        policy_agent.record_reward(-1.0)
    elif winner == -1:
        policy_agent.record_reward(0.0)

    policy_agent.finish_episode()
    return winner


def train_policy_mixed(
    episodes=1000,
    board_size=9,
    device=None,
    opponent_policy_path=MODEL_DIR / "policy_vs_longest.pth",
    policy_color="black",
):
    """複数の相手とランダムに対戦させて PolicyAgent を学習する"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = GomokuEnv(board_size=board_size)
    policy_agent = PolicyAgent(board_size=board_size, device=device)

    # 相手エージェントを用意
    heuristic_agent = LongestChainAgent()
    opponent_policy = PolicyAgent(
        board_size=board_size, device=device, network_type="dense"
    )
    opponent_policy.load_model(opponent_policy_path)

    wins = []
    for epi in range(episodes):
        # ランダムに対戦相手を選択
        if random.random() < 0.5:
            opponent = heuristic_agent
        else:
            opponent = opponent_policy

        winner = run_episode(env, policy_agent, opponent, policy_color)
        wins.append(1 if (policy_color == "black" and winner == 1) or (
            policy_color == "white" and winner == 2
        ) else 0)

        if (epi + 1) % 100 == 0:
            win_rate = sum(wins[-100:]) / 100
            print(f"Episode {epi+1}, 最近100回の勝率: {win_rate:.2f}")

    save_path = MODEL_DIR / "policy_mixed.pth"
    policy_agent.save_model(save_path)
    print(f"学習済みモデルを {save_path} に保存しました")

    # 各相手に対する最終勝率を簡易評価
    wr_h = evaluate_model(
        policy_path=save_path,
        opponent_agent=LongestChainAgent(),
        num_episodes=200,
        board_size=board_size,
        device=device,
        policy_color=policy_color,
        network_type="conv",
    )
    wr_p = evaluate_model(
        policy_path=save_path,
        opponent_agent=opponent_policy,
        num_episodes=200,
        board_size=board_size,
        device=device,
        policy_color=policy_color,
        network_type="conv",
    )
    print(f"LongestChainAgent への勝率: {wr_h:.2f}")
    print(f"既存 PolicyAgent への勝率: {wr_p:.2f}")

    return policy_agent


def main():
    parser = argparse.ArgumentParser(
        description="ヒューリスティックと既存PolicyAgentに交互に対戦する学習"
    )
    parser.add_argument("--episodes", type=int, default=1000, help="学習エピソード数")
    parser.add_argument("--board-size", type=int, default=9, help="盤面サイズ")
    parser.add_argument(
        "--policy-color",
        choices=["black", "white"],
        default="black",
        help="PolicyAgent を先手か後手のどちらで学習するか",
    )
    parser.add_argument(
        "--opponent-policy",
        default=str(MODEL_DIR / "policy_vs_longest.pth"),
        help="対戦相手となる既存Policyモデルのパス",
    )
    parser.add_argument("--device", default=None, help="使用デバイス(cuda/cpu)")
    args = parser.parse_args()

    train_policy_mixed(
        episodes=args.episodes,
        board_size=args.board_size,
        device=args.device,
        opponent_policy_path=Path(args.opponent_policy),
        policy_color=args.policy_color,
    )


if __name__ == "__main__":
    main()
