# -*- coding: utf-8 -*-
"""QAgent をヒューリスティックエージェント相手に段階的に学習させるスクリプト

parallel_q_train.train_master_q() を呼び出し、相手クラスを変えながら複数フェーズ
に渡って学習を行う。各フェーズの終了時には勝率と報酬の平均を表示し、
簡易的なプラトー判定により学習を早期終了することもある。
最終的に play_utils.play_game_text() を用いて 1 戦だけ対局を再現し、
盤面を ASCII 表示する。

盤面サイズは ``--board-size`` オプションで変更できる。5×5 など小さめにすると
学習時間を短縮でき、15×15 のように大きめにするとより実戦的な検証が可能。
各フェーズのエピソード数も ``--episodes`` で調整でき、少ない値で素早く動作確認を、
大きい値では安定した学習を期待できる。
"""

from __future__ import annotations

import argparse
import torch
from pathlib import Path
from typing import Type

from .parallel_q_train import train_master_q
from .play_utils import play_game_text
import multiprocessing
from tqdm import tqdm
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

# check_interval のデフォルト値を定数として定義
# CLI と関数の両方で同じ値を使うことで設定ミスを防ぐ
DEFAULT_CHECK_INTERVAL = 100


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
    show_progress: bool = True,
    check_interval: int = DEFAULT_CHECK_INTERVAL,
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
        show_progress: 進捗バーを表示するかどうか
        check_interval: 勝率を確認するエピソード間隔
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

    phase_iter = enumerate(opponent_classes, start=1)
    if show_progress:
        phase_iter = tqdm(phase_iter, total=len(opponent_classes), desc="Phase")

    for phase, opp in phase_iter:
        print(f"\n==== フェーズ {phase}: 対戦相手 = {opp.__name__} ====")
        phase_win_rates: list[float] = []  # 区間ごとの勝率を蓄積
        phase_rewards: list[float] = []   # フェーズ全体の報酬記録用
        phase_winners: list[int] = []     # 勝敗結果を全て記録
        remaining = episodes_per_phase
        early_stop = False

        # フェーズ中は同じプロセスプールを使い回すことで生成コストを抑える
        # with ブロックを抜けると自動で close される
        with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
            # check_interval ごとに学習を行い、勝率の伸びを逐次確認する
            while remaining > 0:
                cur_eps = min(check_interval, remaining)
                remaining -= cur_eps

                q_agent, rewards, winners, _ = train_master_q(
                    total_episodes=cur_eps,
                    batch_size=max(1, cur_eps // num_workers),
                    board_size=board_size,
                    num_workers=num_workers,
                    agent_params=agent_params,
                    env_params=env_params,
                    opponent_class=opp,
                    show_progress=show_progress,
                    q_agent=q_agent,
                    pool=pool,  # プールを共有して無駄な生成を避ける
                )

                # 各区間の結果を蓄積
                phase_rewards.extend(rewards)
                phase_winners.extend(winners)

                win_rate = sum(1 for w in winners if w == 1) / len(winners)
                avg_reward = sum(rewards) / len(rewards)
                phase_win_rates.append(win_rate)
                win_rates.append(win_rate)
                print(
                    f"区間勝率: {win_rate:.3f}, 区間平均報酬: {avg_reward:.3f}"
                )

                # --- フェーズ内での停滞判定 -------------------------
                if len(phase_win_rates) > plateau_patience:
                    recent = phase_win_rates[-plateau_patience - 1 :]
                    improvement = max(recent) - min(recent)
                    if improvement < plateau_threshold:
                        print("勝率が頭打ちになったためフェーズを早期終了します")
                        early_stop = True
                        if interactive:
                            ans = input("次の相手に進みますか? (y/N): ").strip().lower()
                            if ans not in ("y", "yes"):
                                return q_agent, opp()
                        break

        # with ブロックを抜けるとここでプールが自動的に解放される
        if phase_winners:
            final_win = sum(1 for w in phase_winners if w == 1) / len(phase_winners)
            final_reward = sum(phase_rewards) / len(phase_rewards)
        else:
            final_win = 0.0
            final_reward = 0.0

        print(
            f"フェーズ{phase}終了: 勝率 {final_win:.3f}, 平均報酬 {final_reward:.3f}"
        )

        # --- フェーズ終了後の学習停止判定 ---------------------------
        if final_win >= stop_win_rate:
            print("目標勝率に到達したため学習を終了します")
            break
        if len(win_rates) > plateau_patience:
            recent = win_rates[-plateau_patience - 1 :]
            improvement = max(recent) - min(recent)
            if improvement < plateau_threshold:
                print("勝率が頭打ちと判断したため学習を終了します")
                break

        # フェーズ内で早期終了しなかった場合のみ確認を行う
        if not early_stop and interactive and phase < len(opponent_classes):
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
    # 例: 100 なら素早く動作確認、2000 以上ならじっくり学習
    parser.add_argument("--board-size", type=int, default=9, help="盤面サイズ")
    # 例: 5 を指定すると 5x5 の小盤面で高速に検証できる
    parser.add_argument("--num-workers", type=int, default=4, help="並列ワーカー数")
    parser.add_argument("--device", default=None, help="使用デバイス(cuda/cpu)")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="各フェーズ終了後に次へ進むかを確認する",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="tqdmによる進捗表示を無効化する",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=DEFAULT_CHECK_INTERVAL,
        help="勝率確認を行うエピソード間隔",
    )
    parser.add_argument(
        "--network-type",
        choices=["fc", "conv"],
        default="fc",
        help="QAgent のネットワーク形式",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    agent_params = {"device": device, "network_type": args.network_type}

    q_agent, last_opponent = train_q_vs_heuristics(
        episodes_per_phase=args.episodes,
        board_size=args.board_size,
        num_workers=args.num_workers,
        interactive=args.interactive,
        agent_params=agent_params,
        show_progress=(not args.no_progress),
        check_interval=args.check_interval,
    )

    print("\n=== 学習後の対戦例 ===")
    demo_play(q_agent, last_opponent, board_size=args.board_size)


if __name__ == "__main__":
    main()
