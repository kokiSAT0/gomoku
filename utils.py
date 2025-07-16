# utils.py
"""汎用的なユーティリティ関数をまとめたモジュール"""

from typing import Sequence, List


def moving_average(data: Sequence[float], window: int) -> List[float]:
    """指定された窓幅で移動平均を計算する

    Parameters
    ----------
    data : Sequence[float]
        計算対象となる数値列。
    window : int
        移動平均の窓幅。

    Returns
    -------
    List[float]
        ``data`` と同じ長さの移動平均列を返す。
    """
    # 返却用のリスト
    result: List[float] = []
    # 累積和を保持
    acc = 0.0
    for i, v in enumerate(data):
        acc += v
        if i >= window:
            # 窓を一つ後ろへずらす
            acc -= data[i - window]
            result.append(acc / window)
        else:
            # 要素数がまだ window 未満ならその時点での平均を使う
            result.append(acc / (i + 1))
    return result
