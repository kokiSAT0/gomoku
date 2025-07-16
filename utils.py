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

    # Python のシーケンスを NumPy 配列へ変換
    # "arr" だと意味が分かりにくいので data_array としている
    data_array = np.asarray(data, dtype=float)
    n = data_array.size
    if n == 0:
        return []

    # 累積和(cumsum)を利用して計算量を削減
    cumsum = np.cumsum(data_array)
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
    # 盤面サイズは使わないが分かりやすいよう変数に保持
    board_size = obs.shape[0]
    # 石が置かれていない位置をすべて抽出
    # np.argwhere は条件を満たすインデックスの配列を返す
    empty_positions = np.argwhere(obs == 0)

    valid_actions: list[int] = []
    # 抽出した空きマスを走査し、実際に着手可能かを確認
    for x, y in empty_positions:
        ix = int(x)
        iy = int(y)
        # ルール上石が置ける場合のみ action として登録
        if env.can_place_stone(ix, iy):
            valid_actions.append(env.coord_to_action(ix, iy))

    return valid_actions


def mask_probabilities(probs: np.ndarray, valid_actions: list[int]) -> np.ndarray:
    """無効手の確率を0にして正規化した配列を返す"""
    # 全体を 0 で初期化した配列を用意
    masked_probs = np.zeros_like(probs)

    # 有効な手のみ元の確率をコピーする
    for a in valid_actions:
        masked_probs[a] = probs[a]

    # 0 でない場合は確率の総和で割り正規化
    total = masked_probs.sum()
    if total > 0.0:
        masked_probs /= total

    return masked_probs


def mask_q_values(q_values: np.ndarray, valid_actions: list[int], invalid_value: float = -1e9) -> np.ndarray:
    """無効手のQ値を ``invalid_value`` で置き換える"""
    # 元のQ値を保持したまま加工できるようコピーを作成
    masked_q = q_values.copy()

    # True が無効手を示すブール配列を生成
    invalid_mask = np.ones_like(masked_q, dtype=bool)

    # 許可された手は False にしてマスクを外す
    for a in valid_actions:
        invalid_mask[a] = False

    # マスクされた箇所は非常に小さい値で塗りつぶす
    masked_q[invalid_mask] = invalid_value

    return masked_q


class ReplayBuffer:
    """経験再生用のシンプルなバッファ"""

    def __init__(self, capacity: int = 10000) -> None:
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool) -> None:
        """1ステップ分の遷移を保存"""
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        """ランダムに ``batch_size`` 件取り出し NumPy 配列として返す"""
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        states = np.stack(s).astype(np.float32)
        next_states = np.stack(s_next).astype(np.float32)
        actions = np.array(a, dtype=np.int64)
        rewards = np.array(r, dtype=np.float32)
        dones = np.array(d, dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
