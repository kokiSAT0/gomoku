"""勝ち手を逃さず、相手の勝ち手も防ぐエージェント"""

import random
from pathlib import Path

from ...core.utils import opponent_player, get_valid_actions, find_chain_move

# モデル保存先 (主にインタフェース維持のためのダミー)
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_DIR.mkdir(exist_ok=True)


class ImmediateWinBlockAgent:
    """即勝ち手とブロック手を最優先で選択する"""

    def get_action(self, obs, env):
        """盤面を見て最適と思われる一手を返す"""
        current = env.current_player
        opponent = opponent_player(current)

        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0  # 着手できなければパス

        # 1. 自分が即座に勝てる手を探す
        move = find_chain_move(obs, valid_actions, current, 5)
        if move is not None:
            return move

        # 2. 相手が勝ちそうな手をブロック
        move = find_chain_move(obs, valid_actions, opponent, 5)
        if move is not None:
            return move

        # 3. 残りはランダムに選択
        return random.choice(valid_actions)

    def record_transition(self, s, a, r, s_next, done):
        """学習処理は行わない"""
        pass

    def finish_episode(self):
        pass

    def save_model(self, path=MODEL_DIR / "immediate_win_block_agent.pth"):
        pass

    def load_model(self, path=MODEL_DIR / "immediate_win_block_agent.pth"):
        pass


__all__ = ["ImmediateWinBlockAgent"]
