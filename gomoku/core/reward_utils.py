# -*- coding: utf-8 -*-
"""報酬計算に関するユーティリティ関数群"""

from __future__ import annotations

import numpy as np

from .game import count_chains_open_ends, Gomoku


def calc_chain_reward(before: dict, after: dict, table: dict) -> float:
    """連数の変化量に応じた報酬を計算するヘルパー"""
    reward = 0.0
    for length in [2, 3, 4]:
        for open_type in ["open2", "open1"]:
            diff = after[length][open_type] - before[length][open_type]
            if diff > 0:
                reward += diff * table[length][open_type]
    return reward


def calculate_intermediate_reward(
    board: np.ndarray,
    current_player: int,
    opponent: int,
    before_self: dict,
    before_opp: dict,
    r_chain: dict,
    r_block: dict,
) -> float:
    """盤面の変化から中間報酬を算出する"""
    after_self = count_chains_open_ends(board, current_player)
    after_opp = count_chains_open_ends(board, opponent)

    reward = calc_chain_reward(before_self, after_self, r_chain)
    reward += calc_chain_reward(after_opp, before_opp, r_block)
    return reward


def final_reward(winner: int, current_player: int) -> float:
    """勝敗に基づく最終報酬を返す"""
    if winner == 1:
        return 1.0 if current_player == 1 else -1.0
    if winner == 2:
        return 1.0 if current_player == 2 else -1.0
    return 0.0


def compute_rewards(
    game: Gomoku,
    current_player: int,
    opponent: int,
    before_self: dict,
    before_opp: dict,
    r_chain: dict,
    r_block: dict,
) -> tuple[float, bool, int]:
    """勝敗判定と中間報酬計算を合わせて行う"""
    winner = game.check_winner()
    reward_final = 0.0
    done = False
    if winner != 0:
        done = True
        reward_final = final_reward(winner, current_player)
    reward_intermediate = 0.0
    if not done:
        reward_intermediate = calculate_intermediate_reward(
            game.board,
            current_player,
            opponent,
            before_self,
            before_opp,
            r_chain,
            r_block,
        )
    return reward_final + reward_intermediate, done, winner
