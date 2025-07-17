# -*- coding: utf-8 -*-
"""学習実行の汎用処理をまとめたモジュール"""

# 各スクリプトで重複していた処理を関数化し、再利用しやすくするための
# ラッパー関数群。ここでは ``QAgent`` 同士の学習例と、
# 学習済みモデルを用いた対戦実行を提供する。

from pathlib import Path

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import QAgent
from .learning_utils import train_agents, plot_results, run_match_pygame

# モデル保存先ディレクトリ
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)


def train_q_vs_q(config: dict, show_plot: bool = False) -> tuple[QAgent, QAgent]:
    """QAgent 同士を自己対戦させて学習させる簡易関数"""

    # 環境を作成
    env = GomokuEnv(board_size=config["board_size"], **config["env_params"])

    # 黒番・白番とも同じハイパーパラメータで初期化
    black_q = QAgent(**config["q_params"])
    white_q = QAgent(**config["q_params"])

    # 実際の学習ループは ``train_agents`` に任せる
    rew_b, rew_w, winners, turns = train_agents(
        env,
        black_q,
        white_q,
        episodes=config["episodes"],
    )

    # 成果をグラフ化し必要なら表示
    plot_results(
        rew_b,
        rew_w,
        winners,
        turns,
        title="Q vs Q",
        show=show_plot,
    )

    # 学習済みモデルを保存
    black_q.save_model(MODEL_DIR / "q_black.pth")
    white_q.save_model(MODEL_DIR / "q_white.pth")

    return black_q, white_q


def load_trained_q_agents(config: dict) -> tuple[QAgent, QAgent]:
    """保存済みの QAgent モデルを読み込んで返す"""

    black_q = QAgent(**config["q_params"])
    black_q.load_model(MODEL_DIR / "q_black.pth")

    white_q = QAgent(**config["q_params"])
    white_q.load_model(MODEL_DIR / "q_white.pth")

    return black_q, white_q


def play_trained_match(config: dict, pause_time: float = 0.5) -> None:
    """保存済みモデル同士を PyGame で 1 試合だけ対戦させる"""

    black_q, white_q = load_trained_q_agents(config)
    run_match_pygame(
        black_q,
        white_q,
        board_size=config["board_size"],
        pause_time=pause_time,
        env_params=config["env_params"],
    )


__all__ = ["train_q_vs_q", "load_trained_q_agents", "play_trained_match"]
