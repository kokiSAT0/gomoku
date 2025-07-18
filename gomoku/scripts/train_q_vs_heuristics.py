# -*- coding: utf-8 -*-
"""QAgent をヒューリスティックエージェント相手に段階的に学習させるスクリプト

parallel_q_train.train_master_q() を呼び出し、相手クラスを変えながら複数フェーズ
に渡って学習を行う。各フェーズの終了時には勝率と報酬の平均を表示し、
簡易的なプラトー判定により学習を早期終了することもある。
最終的に play_utils.play_game_text() を用いて 1 戦だけ対局を再現し、
盤面を ASCII 表示する。
"""

from __future__ import annotations

import argparse
import torch
from pathlib import Path
from typing import Type

from .parallel_q_train import train_master_q
from .play_utils import play_game_text
from ..core.gomoku_env import GomokuEnv
from ..ai.agents import (
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent,
)

# モデル保存ディレクトリ
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)


def train_q_vs_heuristics(
    episodes_per_phase: int = 500,
    board_size: int = 9,
    num_workers: int = 4,
    env_params: dict | None = None,
    agent_params: dict | None = None,
    opponent_classes: list[Type] | None = None,
    plateau_threshold: float = 0.01,
    plateau_patience: int = 2,
    stop_win_rate: float = 0.9,
    interactive: bool = False,
):
    """複数のヒューリスティック相手に QAgent を順番に学習させる

    Args:
        episodes_per_phase: 各フェーズの試行回数
        board_size: 盤面の大きさ
        num_workers: 並列ワーカー数
        env_params: 環境に渡す追加パラメータ
        agent_params: エージェント作成時のパラメータ
            ("device" キーで GPU/CPU を指定可能)
        opponent_classes: 対戦相手となるエージェントクラスのリスト
        plateau_threshold: 勝率向上が停滞したとみなす閾値
        plateau_patience: 停滞を確認する期間
        stop_win_rate: 学習終了と判断する勝率
        interactive: True の場合、各フェーズ終了時に続行するか確認する
    """

    if env_params is None:
        env_params = {}
    if agent_params is None:
        agent_params = {}
    if opponent_classes is None:
        opponent_classes = [
            RandomAgent,
            ImmediateWinBlockAgent,
            FourThreePriorityAgent,
            LongestChainAgent,
        ]

    q_agent = None
    win_rates: list[float] = []

    for phase, opp in enumerate(opponent_classes, start=1):
        print(f"\n==== フェーズ {phase}: 対戦相手 = {opp.__name__} ====")
        q_agent, rewards, winners, _ = train_master_q(
            total_episodes=episodes_per_phase,
            batch_size=max(1, episodes_per_phase // num_workers),
            board_size=board_size,
            num_workers=num_workers,
            agent_params=agent_params,
            env_params=env_params,
            opponent_class=opp,
        )

        win_rate = sum(1 for w in winners if w == 1) / len(winners)
        avg_reward = sum(rewards) / len(rewards)
        win_rates.append(win_rate)
        print(f"勝率: {win_rate:.3f}, 平均報酬: {avg_reward:.3f}")

        # --- 学習停止判定 -------------------------------------------
        if win_rate >= stop_win_rate:
            print("目標勝率に到達したため学習を終了します")
            break
        if len(win_rates) > plateau_patience:
            recent = win_rates[-plateau_patience - 1 :]
            improvement = max(recent) - min(recent)
            if improvement < plateau_threshold:
                print("勝率が頭打ちと判断したため学習を終了します")
                break

        # --- インタラクティブモード -------------------------------
        if interactive and phase < len(opponent_classes):
            ans = input("次の相手に進みますか? (y/N): ").strip().lower()
            if ans not in ("y", "yes"):
                break

    return q_agent, opp()


def demo_play(q_agent, opponent_agent, board_size: int = 9) -> None:
    """学習後のエージェント同士で 1 戦だけ対局し盤面を表示"""

    env = GomokuEnv(board_size=board_size)
    play_game_text(env, q_agent, opponent_agent, pause=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="QAgent 段階学習デモ")
    parser.add_argument("--episodes", type=int, default=500, help="各フェーズのエピソード数")
    parser.add_argument("--board-size", type=int, default=9, help="盤面サイズ")
    parser.add_argument("--num-workers", type=int, default=4, help="並列ワーカー数")
    parser.add_argument("--device", default=None, help="使用デバイス(cuda/cpu)")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="各フェーズ終了後に次へ進むかを確認する",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    agent_params = {"device": device}

    q_agent, last_opponent = train_q_vs_heuristics(
        episodes_per_phase=args.episodes,
        board_size=args.board_size,
        num_workers=args.num_workers,
        interactive=args.interactive,
        agent_params=agent_params,
    )

    print("\n=== 学習後の対戦例 ===")
    demo_play(q_agent, last_opponent, board_size=args.board_size)


if __name__ == "__main__":
    main()
