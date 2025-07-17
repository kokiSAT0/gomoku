# -*- coding: utf-8 -*-
"""五目並べの盤面管理クラスと補助関数群"""

import numpy as np

from .utils import opponent_player


def _scan_chain_open_end(
    board: np.ndarray,
    x: int,
    y: int,
    dx: int,
    dy: int,
    player: int,
):
    """指定座標から1方向に伸びる連の長さと両端の空き状況を返すヘルパー"""

    board_size = board.shape[0]

    # 逆方向に同じ色の石がある場合、ここは連の始点ではない
    prev_x, prev_y = x - dx, y - dy
    if 0 <= prev_x < board_size and 0 <= prev_y < board_size:
        if board[prev_x, prev_y] == player:
            return 0, 0

    # (x, y) から (dx, dy) 方向に連を伸ばす
    length = 1
    cx, cy = x + dx, y + dy
    while 0 <= cx < board_size and 0 <= cy < board_size and board[cx, cy] == player:
        length += 1
        cx += dx
        cy += dy

    # 両端が空いているかを確認
    open_ends = 0
    if 0 <= prev_x < board_size and 0 <= prev_y < board_size:
        if board[prev_x, prev_y] == 0:
            open_ends += 1
    if 0 <= cx < board_size and 0 <= cy < board_size:
        if board[cx, cy] == 0:
            open_ends += 1

    return length, open_ends


def count_chains_open_ends(board: np.ndarray, player: int):
    """連の長さ別に両端の空き状況を調べ、本数を辞書で返す"""

    board_size = board.shape[0]
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    result = {
        2: {"open2": 0, "open1": 0},
        3: {"open2": 0, "open1": 0},
        4: {"open2": 0, "open1": 0},
    }

    for x in range(board_size):
        for y in range(board_size):
            if board[x, y] != player:
                continue

            for dx, dy in directions:
                length, open_ends = _scan_chain_open_end(board, x, y, dx, dy, player)

                if length in [2, 3, 4]:
                    if open_ends == 2:
                        result[length]["open2"] += 1
                    elif open_ends == 1:
                        result[length]["open1"] += 1

    return result


class Gomoku:
    """五目並べの基本ロジックを扱うクラス"""

    def __init__(self, board_size: int = 9) -> None:
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 先手=1(黒), 後手=2(白)

    def reset(self) -> None:
        """盤面をリセットして先手を黒番(1)にする"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1

    def is_valid_move(self, x: int, y: int) -> bool:
        """座標が盤内かつ空きマスなら ``True``"""
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False
        return self.board[x, y] == 0

    def place_stone(self, x: int, y: int) -> None:
        """指定座標に現在のプレイヤーの石を置いて手番を交代"""
        self.board[x, y] = self.current_player
        self.current_player = opponent_player(self.current_player)

    def check_winner(self) -> int:
        """勝者を判定する。黒勝ち=1、白勝ち=2、引き分け=-1、続行=0"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for x in range(self.board_size):
            for y in range(self.board_size):
                player = self.board[x, y]
                if player == 0:
                    continue
                for dx, dy in directions:
                    if self._count_stones(x, y, dx, dy, player) >= 5:
                        return player
        if np.all(self.board != 0):
            return -1
        return 0

    def _count_stones(self, x: int, y: int, dx: int, dy: int, player: int) -> int:
        """1方向に連続する同色の石の数を数える"""
        count = 0
        cx, cy = x, y
        while 0 <= cx < self.board_size and 0 <= cy < self.board_size:
            if self.board[cx, cy] == player:
                count += 1
                cx += dx
                cy += dy
            else:
                break
        return count
