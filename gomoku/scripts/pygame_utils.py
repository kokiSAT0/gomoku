"""PyGame 表示関連のヘルパー関数をまとめたモジュール"""

import pygame
import numpy as np

# 描画に使用する色やサイズの定数
CELL_SIZE = 50                # 1マスのピクセル数
LINE_COLOR = (0, 0, 0)        # グリッド線の色
BG_COLOR = (222, 184, 135)    # 碁盤の背景色
BLACK_STONE_COLOR = (0, 0, 0) # 黒石の色
WHITE_STONE_COLOR = (255, 255, 255) # 白石の色


def draw_grid(screen: pygame.Surface, board_size: int) -> None:
    """碁盤のグリッド線を描画する"""
    for i in range(board_size):
        # 縦線の描画
        start = (i * CELL_SIZE + CELL_SIZE // 2, CELL_SIZE // 2)
        end = (i * CELL_SIZE + CELL_SIZE // 2,
               (board_size - 1) * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.line(screen, LINE_COLOR, start, end, 2)

        # 横線の描画
        start = (CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2)
        end = ((board_size - 1) * CELL_SIZE + CELL_SIZE // 2,
               i * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.line(screen, LINE_COLOR, start, end, 2)


def draw_stones(screen: pygame.Surface, board: np.ndarray) -> None:
    """盤面上の石をすべて描画する"""
    for x, y in np.argwhere(board != 0):
        stone = board[x, y]
        center = (y * CELL_SIZE + CELL_SIZE // 2,
                  x * CELL_SIZE + CELL_SIZE // 2)
        color = BLACK_STONE_COLOR if stone == 1 else WHITE_STONE_COLOR
        pygame.draw.circle(screen, color, center, CELL_SIZE // 2 - 2)


def draw_board(screen: pygame.Surface, env) -> None:
    """背景・グリッド・石をまとめて描画する"""
    board_size = env.board_size
    screen.fill(BG_COLOR)
    draw_grid(screen, board_size)
    draw_stones(screen, env.game.board)

