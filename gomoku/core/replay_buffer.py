"""経験再生用バッファ ``ReplayBuffer`` を提供するモジュール"""

# 標準ライブラリ
import random
import collections
from typing import Deque
import numpy as np


class ReplayBuffer:
    """シンプルな経験再生バッファ"""

    def __init__(self, capacity: int = 10000) -> None:
        # 最大容量 ``capacity`` を超えると古いデータから自動的に削除される
        self.buffer: Deque = collections.deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool) -> None:
        """1 ステップ分の遷移を追加する"""
        # ``s`` と ``s_next`` は盤面を表す ``(board_size, board_size)`` 形状を想定
        self.buffer.append((s, a, r, s_next, done))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """蓄積したデータからランダムに ``batch_size`` 件取り出す"""
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        states = np.stack(s).astype(np.float32)
        next_states = np.stack(s_next).astype(np.float32)
        actions = np.array(a, dtype=np.int64)
        rewards = np.array(r, dtype=np.float32)
        dones = np.array(d, dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
