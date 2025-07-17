# -*- coding: utf-8 -*-
"""学習用スクリプトで使うデフォルト設定をまとめたモジュール"""

# ``learning_all_in_one.py`` などで共通利用する設定を一箇所に集約する。
# 実験用に値を調整したい場合はこのファイルを編集するだけで済む。

DEFAULT_CONFIG = {
    "board_size": 9,
    "episodes": 2000,
    "env_params": {
        "force_center_first_move": False,
        "adjacency_range": None,  # None で制限なし
        "invalid_move_penalty": -1.0,
        "reward_chain_2_open2": 0.01,
        "reward_chain_3_open2": 0.5,
        "reward_chain_4_open2": 0.8,
        "reward_chain_2_open1": 0.0,
        "reward_chain_3_open1": 0.05,
        "reward_chain_4_open1": 0.4,
        "reward_block_2_open2": 0.05,
        "reward_block_3_open2": 0.6,
        "reward_block_4_open2": 0.0,
        "reward_block_2_open1": 0.0,
        "reward_block_3_open1": 0.05,
        "reward_block_4_open1": 0.9,
    },
    "policy_params": {
        "board_size": 9,
        "hidden_size": 128,
        "lr": 1e-3,
        "gamma": 0.95,
        "temp": 2.0,
        "min_temp": 0.5,
        "temp_decay": 0.999,
        "entropy_coef": 0.01,
    },
    "q_params": {
        "board_size": 9,
        "hidden_size": 256,
        "lr": 1e-3,
        "gamma": 0.90,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 20000,
        "replay_capacity": 100000,
        "batch_size": 64,
        "update_frequency": 10,
        "target_update_frequency": 200,
    }
}

__all__ = ["DEFAULT_CONFIG"]
