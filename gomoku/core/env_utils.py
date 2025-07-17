"""GomokuEnv で使用する補助関数をまとめたモジュール"""

import numpy as np


def _is_adjacent_to_stone(env, x: int, y: int, rng: int) -> bool:
    """指定位置が既存の石に隣接しているかを判定する内部関数"""
    board = env.game.board
    board_size = env.board_size

    # --- チェビシェフ距離 rng 以内の範囲を計算 ---------------------
    x_min = max(x - rng, 0)
    x_max = min(x + rng + 1, board_size)
    y_min = max(y - rng, 0)
    y_max = min(y + rng + 1, board_size)

    # --- 部分盤面に石が存在するかを numpy で一気に判定 --------------
    sub_board = board[x_min:x_max, y_min:y_max]
    return np.any(sub_board != 0)


def can_place_stone(env, x: int, y: int) -> bool:
    """環境ルールに基づき (x, y) への着手が妥当か判定する"""
    # 1) 初手を中央へ強制する場合のチェック
    if env.turn_count == 0 and env.force_center_first_move:
        center = env.board_size // 2
        if not (x == center and y == center):
            return False

    # 2) adjacency_range チェック (必要ならコメントアウトを外す)
    # if (env.turn_count >= 1) and (env.adjacency_range is not None) and (env.adjacency_range > 0):
    #     if not _is_adjacent_to_stone(env, x, y, env.adjacency_range):
    #         return False

    # 3) 盤外または既に石が置かれていないか
    if not env.game.is_valid_move(x, y):
        return False

    return True

