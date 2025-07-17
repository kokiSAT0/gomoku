"""Pygame を使った簡易対戦スクリプト"""

# 主要処理は play_utils.py にまとめており、ここではサンプルとして呼び出すだけ

from .play_utils import play_agents_vs_agents, MODEL_DIR


if __name__ == "__main__":
    """サンプル実行用エントリポイント"""

    # 黒番 : 学習済みPolicyAgent
    # 白番 : 学習済みPolicyAgent
    play_agents_vs_agents(
        board_size=9,
        num_games=1,
        env_params={"force_center_first_move": False, "adjacency_range": 1},
        black_agent_type="policy",
        black_agent_path=MODEL_DIR / "policy_agent_black.pth",
        black_agent_params={
            "hidden_size": 128,
            "lr": 1e-3,
            "gamma": 0.95,
            "network_type": "dense",
        },
        white_agent_type="policy",
        white_agent_path=MODEL_DIR / "policy_agent_white.pth",
        white_agent_params={"network_type": "dense"},
        visualize=True,
        fps=2,
    )

    # 追加の対戦例を試す場合は下記のコメントを参考にしてください
    # play_agents_vs_agents(
    #     board_size=9,
    #     num_games=5,
    #     env_params={"force_center_first_move": True, "adjacency_range": 1},
    #     black_agent_type="q",
    #     black_agent_path=MODEL_DIR / "q_agent_black.pth",
    #     black_agent_params={"hidden_size": 128, "lr": 1e-3, "gamma": 0.95},
    #     white_agent_type="longest",
    #     white_agent_path=None,
    #     white_agent_params={},
    #     visualize=True,
    #     fps=2,
    # )

