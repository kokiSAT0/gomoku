"""旧 ``heuristic_agents`` モジュールの互換ラッパー"""

# 新しいサブパッケージから同名クラスを再輸入して公開する
from .heuristic import (
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent,
)

__all__ = [
    "RandomAgent",
    "ImmediateWinBlockAgent",
    "FourThreePriorityAgent",
    "LongestChainAgent",
]
