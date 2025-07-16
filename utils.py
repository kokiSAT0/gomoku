# utils.py
"""汎用的なユーティリティ関数をまとめたモジュール"""

from typing import Sequence, List
import random
import collections
import numpy as np


def moving_average(data: Sequence[float], window: int) -> List[float]:
    """指定された窓幅で移動平均を計算する

    Parameters
    ----------
    data : Sequence[float]
        計算対象となる数値列。
    window : int
        移動平均の窓幅。正の整数でなければならない。

    Returns
    -------
    List[float]
        ``data`` と同じ長さの移動平均列を返す。
    """

    if window <= 0:
        raise ValueError("window must be positive")

    # Pythonのリストなどを numpy 配列へ変換
    arr = np.asarray(data, dtype=float)
    n = arr.size
    if n == 0:
        return []

    # 累積和(cumsum)を利用して計算量を削減
    cumsum = np.cumsum(arr)
    # 結果格納用の配列を用意
    result = np.empty(n, dtype=float)

    if n <= window:
        # シンプルに先頭からの平均のみ
        result[:] = cumsum / (np.arange(n) + 1)
        return result.tolist()

    # まず先頭 window 要素分は部分平均
    result[:window] = cumsum[:window] / (np.arange(window) + 1)

    # 以降は前後差分で window 長の和を求めて平均
    result[window:] = (cumsum[window:] - cumsum[:-window]) / window

    return result.tolist()


def opponent_player(player: int) -> int:
    """与えられたプレイヤーID(1 or 2)の相手プレイヤーIDを返す"""
    # 1なら2、2なら1を返すだけのシンプルな関数
    return 2 if player == 1 else 1

# ------------------------------------------------------------
# 盤面関連のユーティリティ
# ------------------------------------------------------------

def get_valid_actions(obs: np.ndarray, env) -> list[int]:
    """盤面から着手可能な action を列挙する"""
    board_size = obs.shape[0]
    empty_positions = np.argwhere(obs == 0)
    valid_actions: list[int] = []
    for x, y in empty_positions:
        ix = int(x)
        iy = int(y)
        if env.can_place_stone(ix, iy):
            valid_actions.append(env.coord_to_action(ix, iy))
    return valid_actions


def mask_probabilities(probs: np.ndarray, valid_actions: list[int]) -> np.ndarray:
    """無効手の確率を0にして正規化した配列を返す"""
    masked = np.zeros_like(probs)
    for a in valid_actions:
        masked[a] = probs[a]
    total = masked.sum()
    if total > 0.0:
        masked /= total
    return masked


def mask_q_values(q_values: np.ndarray, valid_actions: list[int], invalid_value: float = -1e9) -> np.ndarray:
    """無効手のQ値を ``invalid_value`` で置き換える"""
    masked = q_values.copy()
    mask = np.ones_like(masked, dtype=bool)
    for a in valid_actions:
        mask[a] = False
    masked[mask] = invalid_value
    return masked


class ReplayBuffer:
    """経験再生用のシンプルなバッファ

    ゲーム中に得られる状態 ``s``、行動 ``a``、報酬 ``r`` などの遷移を
    一時的に蓄積し、学習時にランダムサンプリングしてミニバッチとして
    取り出す用途を想定している。典型的な強化学習ループでは、各ステップ
    ごとに :meth:`push` で追加し、学習関数内で :meth:`sample` を呼び出す。
    """

    def __init__(self, capacity: int = 10000) -> None:
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool) -> None:
        """1ステップ分の遷移を保存"""
        # s, s_next は (board_size, board_size) の盤面配列を想定
        # 例: board_size=9 の場合は (9, 9) の二次元配列
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        """ランダムに ``batch_size`` 件取り出し NumPy 配列として返す"""
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        # states, next_states: (batch_size, board_size, board_size)
        states = np.stack(s).astype(np.float32)
        next_states = np.stack(s_next).astype(np.float32)
        actions = np.array(a, dtype=np.int64)
        rewards = np.array(r, dtype=np.float32)
        dones = np.array(d, dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
