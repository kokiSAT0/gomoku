"""一般的なユーティリティ関数をまとめたモジュール"""

# 標準ライブラリ
from typing import Sequence, List
from pathlib import Path

# サードパーティ
import numpy as np

# ------------------------------------------------------------
# 図を保存するディレクトリ
#   - GUI が使えない環境でも学習状況を確認できるよう、
#     生成したグラフ画像をここへ集約する
# ------------------------------------------------------------
FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def moving_average(data: Sequence[float], window: int) -> List[float]:
    """移動平均を計算して ``list`` で返す

    Parameters
    ----------
    data : Sequence[float]
        計算対象となる数値列
    window : int
        移動平均に用いる窓幅
    """

    if window <= 0:
        raise ValueError("window must be positive")

    # 元データを numpy 配列へ変換して計算を高速化
    data_array = np.asarray(data, dtype=float)
    n = data_array.size
    if n == 0:
        return []

    # 累積和を利用すると O(n) で計算できる
    cumsum = np.cumsum(data_array)
    result = np.empty(n, dtype=float)

    if n <= window:
        # データ数が窓幅以下なら単純平均のみ
        result[:] = cumsum / (np.arange(n) + 1)
        return result.tolist()

    # 先頭 ``window`` 要素分は部分平均
    result[:window] = cumsum[:window] / (np.arange(window) + 1)

    # 差分を取れば常に ``window`` 長の和が得られる
    result[window:] = (cumsum[window:] - cumsum[:-window]) / window

    return result.tolist()
