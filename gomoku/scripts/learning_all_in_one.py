"""ゲーム環境から学習ループまでを一つにまとめた実験用スクリプト。

五目並べ環境の実装と強化学習によるエージェント学習を
このファイルだけで確認できる。
"""

from pathlib import Path

from ..core.gomoku_env import GomokuEnv
from ..ai.agents import (
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent,
    PolicyAgent,
    QAgent,
)
from .learning_utils import train_agents, plot_results, run_match_pygame

# 学習済みモデルを保存するディレクトリ
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# ------------------------------------------------------------
# グラフ画像を保存するディレクトリ
#   実験結果の可視化を GUI 無し環境でも確認できるよう、
#   ここにまとめて保存する
# ------------------------------------------------------------

def main():
    # ------------------------------
    # 例: ハイパーパラメータ設定
    # ------------------------------
    config = {
        "board_size": 9,
        "episodes": 2000,
        "env_params": {
            "force_center_first_move": False,
            "adjacency_range": None,  # None で制限なし
            "invalid_move_penalty": -1.0,
            "reward_chain_2_open2": 0.01,
            "reward_chain_3_open2": 0.5,
            "reward_chain_4_open2": 0.8,
            "reward_chain_2_open1": 0.0,
            "reward_chain_3_open1": 0.05,
            "reward_chain_4_open1": 0.4,
            "reward_block_2_open2": 0.05,
            "reward_block_3_open2": 0.6,
            "reward_block_4_open2": 0.0,
            "reward_block_2_open1": 0.0,
            "reward_block_3_open1": 0.05,
            "reward_block_4_open1": 0.9,
        },
        "policy_params": {
            "board_size": 9,
            "hidden_size": 128,
            "lr": 1e-3,
            "gamma": 0.95,
            "temp": 2.0,
            "min_temp": 0.5,
            "temp_decay": 0.999,
            "entropy_coef": 0.01,
        },
        "q_params": {
            "board_size": 9,
            "hidden_size": 256,
            "lr": 1e-3,
            "gamma": 0.90,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 20000,
            "replay_capacity": 100000,
            "batch_size": 64,
            "update_frequency": 10,
            "target_update_frequency": 200,
        }
    }

    # 環境作成
    env = GomokuEnv(board_size=config["board_size"], **config["env_params"])

    # ------------------------------
    # 例1: Policy vs Policy (自己対戦)
    # ------------------------------
    # black_agent = PolicyAgent(**config["policy_params"])
    # white_agent = PolicyAgent(**config["policy_params"])

    # rew_b, rew_w, winners, turns = train_agents(env, black_agent, white_agent, config["episodes"])
    # plot_results(rew_b, rew_w, winners, turns, title="Policy vs Policy")

    # black_agent.save_model(MODEL_DIR / "policy_black.pth")
    # white_agent.save_model(MODEL_DIR / "policy_white.pth")

    # ------------------------------
    # 例2: QAgent vs QAgent (自己対戦)
    # ------------------------------
    black_q = QAgent(**config["q_params"])
    white_q = QAgent(**config["q_params"])

    rew_b, rew_w, winners, turns = train_agents(env, black_q, white_q, config["episodes"])
    # GUI が無い場合を考慮して show=False
    plot_results(
        rew_b,
        rew_w,
        winners,
        turns,
        title="Q vs Q",
        show=False,
    )

    black_q.save_model(MODEL_DIR / "q_black.pth")
    white_q.save_model(MODEL_DIR / "q_white.pth")

    # ------------------------------
    # 例3: ヒューリスティックAgent vs 学習Agent
    #     (黒番: Policy, 白番: ImmediateWinBlock)
    # ------------------------------
    # env = GomokuEnv(board_size=config["board_size"], **config["env_params"])
    # black_policy = PolicyAgent(**config["policy_params"])
    # white_heuristic = LongestChainAgent()

    # rew_b, rew_w, winners, turns = train_agents(env, black_policy, white_heuristic, config["episodes"])
    # plot_results(rew_b, rew_w, winners, turns, title="Policy(Black) vs ImmediateWinBlock(White)")

    # ------------------------------
    # 例4: ヒューリスティックAgent vs 学習Agent
    #     (黒番: QAgent, 白番: FourThreePriority)
    # ------------------------------
    # env = GomokuEnv(board_size=config["board_size"], **config["env_params"])
    # black_q = QAgent(**config["q_params"])
    # white_heuristic = LongestChainAgent()

    # rew_b, rew_w, winners, turns = train_agents(env, black_q, white_heuristic, config["episodes"])
    # plot_results(rew_b, rew_w, winners, turns, title="Q(Black) vs FourThreePriority(White)")


    # 好みに合わせて学習させたい組み合わせを試してみてください。

    return config


if __name__ == "__main__":
    config = main()
    black_q = QAgent(**config["q_params"])
    black_q.load_model(MODEL_DIR / "q_black.pth")

    white_q = QAgent(**config["q_params"])
    white_q.load_model(MODEL_DIR / "q_white.pth")

    run_match_pygame(
        black_q,
        white_q,
        board_size=config["board_size"],
        pause_time=0.5,
        env_params=config["env_params"],
    )

