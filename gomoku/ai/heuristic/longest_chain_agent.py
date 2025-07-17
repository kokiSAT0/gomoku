"""仮置きの評価として最長の連を作る手を選択するエージェント"""

import random
from pathlib import Path

from ...core.utils import get_valid_actions, longest_chain_length

MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_DIR.mkdir(exist_ok=True)


class LongestChainAgent:
    """着手後に最長の連ができる手を狙う単純な戦略"""

    def get_action(self, obs, env):
        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0

        current = env.current_player
        best_score = -1
        best_actions = []

        # 置いたときの連長を評価して最大となる手を収集
        for a in valid_actions:
            x, y = env.action_to_coord(a)
            score = self._evaluate_move(obs, x, y, current)
            if score > best_score:
                best_score = score
                best_actions = [a]
            elif score == best_score:
                best_actions.append(a)

        return random.choice(best_actions)

    def _evaluate_move(self, obs, x, y, player):
        """一時的に石を置いて連の長さを計算"""
        obs[x, y] = player
        score = longest_chain_length(obs, x, y, player)
        obs[x, y] = 0
        return score

    def record_transition(self, s, a, r, s_next, done):
        pass

    def record_reward(self, r):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path=MODEL_DIR / "longest_chain_agent.pth"):
        pass

    def load_model(self, path=MODEL_DIR / "longest_chain_agent.pth"):
        pass


__all__ = ["LongestChainAgent"]
