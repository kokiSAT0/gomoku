# learn_model.py

"""強化学習エージェントを自己対戦させて学習させるためのスクリプト。

PolicyAgent や QAgent の学習例が含まれており、
関数呼び出しを通して各種パラメータを調整できる。
"""

from pathlib import Path
from .selfplay import (
    selfplay_policy_vs_policy,
    selfplay_q_vs_q,
    selfplay_policy_vs_q,
)
# 学習済みモデルを格納するフォルダ
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def main():
    # 例1: PolicyAgent vs PolicyAgent
    black_agent, white_agent, rew_b, rew_w, winners, turns = selfplay_policy_vs_policy(
        board_size=9,
        episodes=4000,
        env_params={"force_center_first_move": False, "adjacency_range": None},
        black_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},
        white_params={"hidden_size":128, "lr":1e-3, "gamma":0.95}
    )
    # GUI のない環境でも確認できるよう show=False
    plot_results(
        rew_b,
        rew_w,
        winners,
        turns,
        title="Policy vs Policy (9x9)",
        show=False,
    )
    black_agent.save_model(MODEL_DIR / "policy_agent_black.pth")
    white_agent.save_model(MODEL_DIR / "policy_agent_white.pth")

    # 例2: QAgent vs QAgent
    # black_agent, white_agent, rew_b, rew_w, winners, turns = selfplay_q_vs_q(
    #     board_size=9,
    #     episodes=3000,
    #     env_params={"force_center_first_move": False, "adjacency_range": None},
    #     black_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},
    #     white_params={"hidden_size":128, "lr":1e-3, "gamma":0.95}
    # )
    # plot_results(rew_b, rew_w, winners, turns, title="Q vs Q (9x9)")
    # black_agent.save_model(MODEL_DIR / "q_agent_black.pth")
    # white_agent.save_model(MODEL_DIR / "q_agent_white.pth")

    # 例3: Policy(黒) vs Q(白)
    # black_agent, white_agent, rew_b, rew_w, winners, turns = selfplay_policy_vs_q(
    #     board_size=9,
    #     episodes=2000,
    #     env_params={"force_center_first_move": False, "adjacency_range": None},
    #     black_is_policy=True,
    #     policy_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},
    #     q_params={"hidden_size":128, "lr":1e-3, "gamma":0.95}
    # )
    # plot_results(rew_b, rew_w, winners, turns, title="Policy(Black) vs Q(White)")
    # black_agent.save_model(MODEL_DIR / "policy_black.pth")
    # white_agent.save_model(MODEL_DIR / "q_white.pth")

    # 例4: Q(黒) vs Policy(白)
    # black_agent, white_agent, rew_b, rew_w, winners, turns = selfplay_policy_vs_q(
    #     board_size=9,
    #     episodes=2000,
    #     env_params={"force_center_first_move": False, "adjacency_range": None},
    #     black_is_policy=False,
    #     policy_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},
    #     q_params={"hidden_size":128, "lr":1e-3, "gamma":0.95}
    # )
    # plot_results(rew_b, rew_w, winners, turns, title="Q(Black) vs Policy(White)")
    # black_agent.save_model(MODEL_DIR / "q_black.pth")
    # white_agent.save_model(MODEL_DIR / "policy_white.pth")


if __name__ == "__main__":
    main()
