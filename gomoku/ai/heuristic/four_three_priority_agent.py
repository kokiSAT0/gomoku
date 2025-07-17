"""四連・三連の形成を優先する基本的な戦略エージェント"""

import random
from pathlib import Path

from ...core.utils import opponent_player, get_valid_actions, find_chain_move

MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_DIR.mkdir(exist_ok=True)


class FourThreePriorityAgent:
    """4連完成 > 4連ブロック > 3連狙い > 3連ブロックの優先度で手を選ぶ"""

    def get_action(self, obs, env):
        current = env.current_player
        opponent = opponent_player(current)

        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0

        # 1. 自分の4連を完成させる手
        move = find_chain_move(obs, valid_actions, current, 4)
        if move is not None:
            return move

        # 2. 相手の4連を阻止
        move = find_chain_move(obs, valid_actions, opponent, 4)
        if move is not None:
            return move

        # 3. 自分の3連を伸ばす
        move = find_chain_move(obs, valid_actions, current, 3)
        if move is not None:
            return move

        # 4. 相手の3連をブロック
        move = find_chain_move(obs, valid_actions, opponent, 3)
        if move is not None:
            return move

        # 5. それ以外はランダム
        return random.choice(valid_actions)

    def record_transition(self, s, a, r, s_next, done):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path=MODEL_DIR / "four_three_priority_agent.pth"):
        pass

    def load_model(self, path=MODEL_DIR / "four_three_priority_agent.pth"):
        pass


__all__ = ["FourThreePriorityAgent"]
