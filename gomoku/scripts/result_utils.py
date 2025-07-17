"""対局結果を整形して表示するヘルパー"""

import numpy as np


def show_results(
    black_agent_type: str,
    black_agent_path,
    white_agent_type: str,
    white_agent_path,
    winners: list,
    black_rewards: list,
    turn_counts: list,
) -> None:
    """複数試合の結果を集計して標準出力へ表示する"""
    num_games = len(winners)
    avg_reward = float(np.mean(black_rewards))
    black_win_rate = float(np.mean([1.0 if w == 1 else 0.0 for w in winners]))
    avg_turn_count = float(np.mean(turn_counts))

    print("============================================================")
    print(f"対局数: {num_games}")
    print(f"黒番 = {black_agent_type}, path={black_agent_path}")
    print(f"白番 = {white_agent_type}, path={white_agent_path}")
    print(f"黒番 平均報酬(視点) : {avg_reward:.3f}")
    print(f"黒番 勝率          : {black_win_rate:.3f}")
    print(f"平均決着手数       : {avg_turn_count:.1f}")
    print("------------------------------------------------------------")

    show_count = min(num_games, 10)
    print(f"(先頭{show_count}ゲームの結果) => (winner, black_reward, turn)")
    for i in range(show_count):
        print(
            f"Game {i+1:2d}: Winner={winners[i]}, "
            f"RewardForBlack={black_rewards[i]}, Turn={turn_counts[i]}"
        )
    print("============================================================")

