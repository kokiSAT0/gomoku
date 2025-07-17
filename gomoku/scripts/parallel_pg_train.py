# parallel_pg_train.py

"""PolicyAgent 用の並列学習をラップする互換モジュール。

本モジュール自体には学習ロジックを実装せず、``pg_train_master`` に
移動した ``train_worker`` と ``train_master`` を再利用する。
古いコードからのインポートを壊さないための薄いラッパーとして存在する。
"""

from .pg_train_master import train_worker, train_master
from .pg_train_utils import play_one_episode, update_with_trajectories

__all__ = [
    "play_one_episode",
    "train_worker",
    "train_master",
    "update_with_trajectories",
]
