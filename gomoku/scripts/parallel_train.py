# -*- coding: utf-8 -*-
"""旧 `parallel_train.py`

巨大化した元ファイルを分割し、PolicyAgent 用と QAgent 用の
処理をそれぞれ別モジュールへ移動した。

- PolicyAgent 向け関数: :mod:`parallel_pg_train`
- QAgent 向け関数: :mod:`parallel_q_train`

既存コードとの互換性のため、主要関数をこのモジュールから
再エクスポートしている。
"""

from .parallel_pg_train import (
    play_one_episode,
    train_worker,
    train_master,
    update_with_trajectories,
)

from .parallel_q_train import (
    play_one_episode_q,
    train_worker_q,
    train_master_q,
)

__all__ = [
    "play_one_episode",
    "train_worker",
    "train_master",
    "update_with_trajectories",
    "play_one_episode_q",
    "train_worker_q",
    "train_master_q",
]
