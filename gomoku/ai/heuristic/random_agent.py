"""ランダムに着手するだけのエージェント"""

# ランダム選択のみを行うシンプルな実装

import random
from pathlib import Path

from ...core.utils import get_valid_actions

# モデル保存用ディレクトリ (現状では使わないが雛形として残す)
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_DIR.mkdir(exist_ok=True)


class RandomAgent:
    """ランダムに合法手を返すエージェント"""

    def get_action(self, obs, env):
        """合法手から一つランダムに選ぶ"""
        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0  # 着手可能なマスが無い場合はパス
        return random.choice(valid_actions)

    def record_transition(self, s, a, r, s_next, done):
        """学習処理は持たないため何もしない"""
        pass

    def finish_episode(self):
        """学習エージェントとの共通インタフェースとして定義"""
        pass

    def save_model(self, path=MODEL_DIR / "random_agent.pth"):
        """モデル保存用 (現状は未使用)"""
        pass

    def load_model(self, path=MODEL_DIR / "random_agent.pth"):
        """モデル読み込み用 (現状は未使用)"""
        pass


__all__ = ["RandomAgent"]
