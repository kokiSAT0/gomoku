# utils.py
"""汎用的なユーティリティ関数をまとめたモジュール"""

from typing import Sequence, List
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
