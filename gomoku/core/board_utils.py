"""盤面操作に関するユーティリティをまとめたモジュール"""

from __future__ import annotations

# 標準ライブラリ
from typing import List
import numpy as np

# ------------------------------------------------------------
# プレイヤーID関連のヘルパー
# ------------------------------------------------------------

def opponent_player(player: int) -> int:
    """相手プレイヤーのIDを返す"""
    # 1なら2、2なら1を返すだけ
    return 2 if player == 1 else 1


# ------------------------------------------------------------
# 合法手の列挙および確率・Q値マスク関連
# ------------------------------------------------------------

def get_valid_actions(obs: np.ndarray, env) -> List[int]:
    """盤面 ``obs`` と環境 ``env`` から着手可能な行動を列挙する"""
    empty_positions = np.argwhere(obs == 0)

    valid_actions: List[int] = []
    for x, y in empty_positions:
        ix = int(x)
        iy = int(y)
        if env.can_place_stone(ix, iy):
            valid_actions.append(env.coord_to_action(ix, iy))
    return valid_actions


def mask_probabilities(probs: np.ndarray, valid_actions: List[int]) -> np.ndarray:
    """無効手を除外して確率分布を再正規化する"""
    masked_probs = np.zeros_like(probs)
    for a in valid_actions:
        masked_probs[a] = probs[a]

    total = masked_probs.sum()
    if total > 0.0:
        masked_probs /= total
    return masked_probs


def mask_q_values(
    q_values: np.ndarray,
    valid_actions: List[int],
    invalid_value: float = -1e9,
) -> np.ndarray:
    """無効手の Q 値を ``invalid_value`` で塗りつぶす"""
    masked_q = q_values.copy()
    invalid_mask = np.ones_like(masked_q, dtype=bool)
    for a in valid_actions:
        invalid_mask[a] = False
    masked_q[invalid_mask] = invalid_value
    return masked_q


# ------------------------------------------------------------
# 連の探索関連
# ------------------------------------------------------------
# 各方向を表す (dx, dy) ペア
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]


def longest_chain_length(
    obs: np.ndarray,
    x: int,
    y: int,
    player: int,
    directions: list[tuple[int, int]] = DIRECTIONS,
) -> int:
    """石を置いたと仮定したときの最長連結長を計算する"""

    board_size = obs.shape[0]
    max_len = 1

    for dx, dy in directions:
        count = 1

        cx, cy = x + dx, y + dy
        while 0 <= cx < board_size and 0 <= cy < board_size and obs[cx, cy] == player:
            count += 1
            cx += dx
            cy += dy

        cx, cy = x - dx, y - dy
        while 0 <= cx < board_size and 0 <= cy < board_size and obs[cx, cy] == player:
            count += 1
            cx -= dx
            cy -= dy

        if count > max_len:
            max_len = count

    return max_len


def has_n_in_a_row(
    obs: np.ndarray,
    x: int,
    y: int,
    player: int,
    n: int,
    directions: list[tuple[int, int]] = DIRECTIONS,
) -> bool:
    """``n`` 連以上が完成するかどうかを判定"""
    return longest_chain_length(obs, x, y, player, directions) >= n


def find_chain_move(
    obs: np.ndarray,
    valid_actions: List[int],
    player: int,
    n: int,
    directions: list[tuple[int, int]] = DIRECTIONS,
) -> int | None:
    """``n`` 連を作れる着手があればその ``action`` を返す"""

    board_size = obs.shape[0]

    for a in valid_actions:
        x = a // board_size
        y = a % board_size

        # 実際に置く前に仮置きしてチェック
        obs[x, y] = player
        if has_n_in_a_row(obs, x, y, player, n, directions):
            obs[x, y] = 0
            return a
        obs[x, y] = 0

    return None
