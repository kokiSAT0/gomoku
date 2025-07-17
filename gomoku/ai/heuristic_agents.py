"""ヒューリスティックベースのエージェント群"""
# ランダム選択や簡易ルールに基づくエージェントを定義する

import random
from pathlib import Path

from ..core.utils import (
    opponent_player,
    get_valid_actions,
    longest_chain_length,
    find_chain_move,
)

# 学習済みモデルの保存先
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)


class RandomAgent:
    """ランダムに着手するだけのエージェント"""

    def get_action(self, obs, env):
        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0  # 打てる場所が無ければパス
        return random.choice(valid_actions)

    def record_transition(self, s, a, r, s_next, done):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path=MODEL_DIR / "random_agent.pth"):
        pass

    def load_model(self, path=MODEL_DIR / "random_agent.pth"):
        pass


class ImmediateWinBlockAgent:
    """勝ち手やブロック手を優先するエージェント"""

    def get_action(self, obs, env):
        current_player = env.current_player
        opponent = opponent_player(current_player)

        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0  # 着手可能な場所がなければパス

        # 勝ち手があるなら即座に打つ
        move = find_chain_move(obs, valid_actions, current_player, 5)
        if move is not None:
            return move

        # 相手の勝ち手があればブロック
        move = find_chain_move(obs, valid_actions, opponent, 5)
        if move is not None:
            return move

        # それ以外はランダムに着手
        return random.choice(valid_actions)

    def record_transition(self, s, a, r, s_next, done):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path=MODEL_DIR / "immediate_win_block_agent.pth"):
        pass

    def load_model(self, path=MODEL_DIR / "immediate_win_block_agent.pth"):
        pass


class FourThreePriorityAgent:
    """4連や3連を優先しつつブロックも行うエージェント"""

    def get_action(self, obs, env):
        current_player = env.current_player
        opponent = opponent_player(current_player)

        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0

        # 自分の4連完成を最優先
        a = find_chain_move(obs, valid_actions, current_player, 4)
        if a is not None:
            return a

        # 次に相手の4連をブロック
        a = find_chain_move(obs, valid_actions, opponent, 4)
        if a is not None:
            return a

        # 続いて自分の3連完成を狙う
        a = find_chain_move(obs, valid_actions, current_player, 3)
        if a is not None:
            return a

        # 最後に相手の3連をブロック
        a = find_chain_move(obs, valid_actions, opponent, 3)
        if a is not None:
            return a

        # どれも該当しなければランダムに着手
        return random.choice(valid_actions)

    def record_transition(self, s, a, r, s_next, done):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path=MODEL_DIR / "four_three_priority_agent.pth"):
        pass

    def load_model(self, path=MODEL_DIR / "four_three_priority_agent.pth"):
        pass


class LongestChainAgent:
    """仮置きで最長の連を作る手を選ぶエージェント"""

    def get_action(self, obs, env):
        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0

        current_player = env.current_player
        best_score = -1
        best_actions = []

        # 置いたときに最長の連ができる手を探す
        for a in valid_actions:
            x, y = env.action_to_coord(a)
            score = self._evaluate_move(obs, x, y, current_player)
            if score > best_score:
                best_score = score
                best_actions = [a]
            elif score == best_score:
                best_actions.append(a)

        return random.choice(best_actions)

    def _evaluate_move(self, obs, x, y, player):
        # 評価のために一時的に石を置く
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
