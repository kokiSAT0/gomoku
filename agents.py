# agents.py

"""各種エージェントクラスを定義するモジュール。

ランダム戦略から強化学習ベースのエージェントまでを実装し、
他のスクリプトからインポートして ``get_action`` を呼び出すことで
対戦や学習に利用できる。
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

from utils import opponent_player, get_valid_actions, mask_probabilities, mask_q_values, ReplayBuffer

# ----------------------------------------------------
# 学習済みモデルを保存するディレクトリ
# ----------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent / "models"
# フォルダが存在しない場合は作成しておく
MODEL_DIR.mkdir(exist_ok=True)


@dataclass  # dataclass を重複して指定する必要はないので一つだけにする
class EpisodeStep:
    """PolicyAgentの1ステップ分の情報を保持するデータクラス"""

    state: torch.Tensor  # 盤面状態 (flatten 済みテンソル)
    action: int          # 選択した行動
    reward: float        # その行動で得た報酬

# ----------------------------------------------------
# 連をチェックする際に利用する方向のリスト
# ----------------------------------------------------
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

# ----------------------------------------------------
# 有効手(空マス)を列挙する関数
# ----------------------------------------------------
def longest_chain_length(obs, x, y, player, directions=DIRECTIONS):
    """指定座標に石を置いたと仮定したときの最長連結長を返すヘルパー"""
    board_size = obs.shape[0]
    max_len = 1
    for dx, dy in directions:
        count = 1
        # 正方向へ伸びている石を数える
        cx, cy = x + dx, y + dy
        while 0 <= cx < board_size and 0 <= cy < board_size and obs[cx, cy] == player:
            count += 1
            cx += dx
            cy += dy
        # 逆方向へも伸ばす
        cx, cy = x - dx, y - dy
        while 0 <= cx < board_size and 0 <= cy < board_size and obs[cx, cy] == player:
            count += 1
            cx -= dx
            cy -= dy
        if count > max_len:
            max_len = count
    return max_len

def has_n_in_a_row(obs, x, y, player, n, directions=DIRECTIONS):
    """n連以上が成立するかどうかを判定"""
    return longest_chain_length(obs, x, y, player, directions) >= n


def find_chain_move(obs, valid_actions, player, n, directions=DIRECTIONS):
    """n連を作れる着手を探して返す。存在しなければNone"""
    board_size = obs.shape[0]
    for a in valid_actions:
        x = a // board_size
        y = a % board_size

        # 仮に石を置いてn連が成立するか確認
        obs[x, y] = player
        if has_n_in_a_row(obs, x, y, player, n, directions):
            obs[x, y] = 0
            return a
        obs[x, y] = 0

    return None

# ----------------------------------------------------
# 方策分布やQ値をマスクするユーティリティ関数
# ----------------------------------------------------




# ----------------------------------------------------
# ランダムエージェント (学習しない)
# ----------------------------------------------------
class RandomAgent:
    """
    ランダムに着手するだけの単純なエージェント。

    学習は一切行わず、合法手の中からランダムに行動を選択する。
    """
    def __init__(self):
        pass

    def get_action(self, obs, env):
        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0
        return random.choice(valid_actions)

    def record_transition(self, s, a, r, s_next, done):
        """このエージェントでは未使用"""
        pass

    def finish_episode(self):
        """このエージェントでは未使用"""
        pass

    def save_model(self, path=MODEL_DIR / "random_agent.pth"):
        """このエージェントでは未使用"""
        pass

    def load_model(self, path=MODEL_DIR / "random_agent.pth"):
        """このエージェントでは未使用"""
        pass


# ----------------------------------------------------
# ヒューリスティックエージェント各種
# ----------------------------------------------------
class ImmediateWinBlockAgent:
    """
    置いて即勝ち手を優先。なければ相手の即勝ち手をブロック。なければランダム。
    """
    def __init__(self):
        pass

    def get_action(self, obs, env):
        current_player = env.current_player
        # 相手プレイヤーIDをユーティリティ関数で取得
        opponent = opponent_player(current_player)

        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0

        # 1) 自分が置けば5連になるか？
        move = find_chain_move(obs, valid_actions, current_player, 5)
        if move is not None:
            return move

        # 2) 相手が次手で5連できるならブロック
        move = find_chain_move(obs, valid_actions, opponent, 5)
        if move is not None:
            return move

        # 3) 上記がなければランダム
        return random.choice(valid_actions)


    def record_transition(self, s, a, r, s_next, done):
        """このエージェントでは未使用"""
        pass

    def finish_episode(self):
        """このエージェントでは未使用"""
        pass

    def save_model(self, path=MODEL_DIR / "immediate_win_block_agent.pth"):
        """このエージェントでは未使用"""
        pass

    def load_model(self, path=MODEL_DIR / "immediate_win_block_agent.pth"):
        """このエージェントでは未使用"""
        pass


class FourThreePriorityAgent:
    """
    4連/3連を優先し、ブロックもする簡易ヒューリスティック
    """
    def __init__(self):
        pass

    def get_action(self, obs, env):
        current_player = env.current_player
        # 相手プレイヤーIDを取得
        opponent = opponent_player(current_player)

        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0

        # 1) 自分の4連が完成する手
        a = find_chain_move(obs, valid_actions, current_player, 4)
        if a is not None:
            return a

        # 2) 相手の4連ブロック
        a = find_chain_move(obs, valid_actions, opponent, 4)
        if a is not None:
            return a

        # 3) 自分の3連が完成する手
        a = find_chain_move(obs, valid_actions, current_player, 3)
        if a is not None:
            return a

        # 4) 相手の3連ブロック
        a = find_chain_move(obs, valid_actions, opponent, 3)
        if a is not None:
            return a

        # 5) ランダム
        return random.choice(valid_actions)


    def record_transition(self, s, a, r, s_next, done):
        """このエージェントでは未使用"""
        pass

    def finish_episode(self):
        """このエージェントでは未使用"""
        pass

    def save_model(self, path=MODEL_DIR / "four_three_priority_agent.pth"):
        """このエージェントでは未使用"""
        pass

    def load_model(self, path=MODEL_DIR / "four_three_priority_agent.pth"):
        """このエージェントでは未使用"""
        pass


class LongestChainAgent:
    """
    盤上に仮置きしたとき最も長い連が得られる手を選択するヒューリスティックエージェント。
    """

    def __init__(self):
        # 特別な内部状態は持たない
        pass

    def get_action(self, obs, env):
        """盤面と環境から行動を決定して返す"""
        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0  # 打てる場所が無い場合は 0 を返す

        current_player = env.current_player
        best_score = -1
        best_actions = []  # 同点ならランダムに選ぶためリストに保持

        for a in valid_actions:
            # action 番号を座標へ変換
            x, y = env.action_to_coord(a)

            # 仮に石を置いて連の長さを評価
            score = self._evaluate_move(obs, x, y, current_player)

            # より良い手なら更新、同点なら追加
            if score > best_score:
                best_score = score
                best_actions = [a]
            elif score == best_score:
                best_actions.append(a)

        return random.choice(best_actions)

    def _evaluate_move(self, obs, x, y, player):
        """石を一時的に置いて連の長さを計算するヘルパー"""
        obs[x, y] = player
        score = longest_chain_length(obs, x, y, player)
        obs[x, y] = 0
        return score

    def record_transition(self, s, a, r, s_next, done):
        """このエージェントでは未使用"""
        pass

    def finish_episode(self):
        """このエージェントでは未使用"""
        pass

    def save_model(self, path=MODEL_DIR / "longest_chain_agent.pth"):
        """このエージェントでは未使用"""
        pass

    def load_model(self, path=MODEL_DIR / "longest_chain_agent.pth"):
        """このエージェントでは未使用"""
        pass


# ----------------------------------------------------
# PolicyAgent (REINFORCE)
# ----------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, board_size=9, hidden_size=128):
        super().__init__()
        input_dim = board_size * board_size
        output_dim = board_size * board_size
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits


class PolicyAgent:
    def __init__(self, board_size=9, hidden_size=128, lr=1e-3, gamma=0.99, temp=2.0):
        """
        - temp: ソフトマックス温度 (初期値)
        """
        self.board_size = board_size
        self.gamma = gamma

        self.model = PolicyNet(board_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # 各ステップの情報を EpisodeStep として蓄積
        self.episode_log: list[EpisodeStep] = []

        self.temp = temp       # 初期温度
        self.min_temp = 0.5    # 温度の下限値など (必要に応じて調整)
        self.temp_decay = 0.999  # エピソード終了ごとに温度を掛け算で減らす例

        self.episode_count = 0  # 何エピソード学習したか

    def get_action(self, obs, env):
        """
        obs: (board_size, board_size) の numpy配列
        env: GomokuEnv
        戻り値: action(int)
        """
        # 1) 盤面をflattenしてテンソル化
        state_t = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)

        # 2) ネットワーク推論
        with torch.no_grad():
            # logits のテンソル形状は (1, board_size*board_size)
            logits = self.model(state_t)
            # ここでソフトマックス温度を適用
            scaled_logits = logits / self.temp
            probs = F.softmax(scaled_logits, dim=1).cpu().numpy().flatten()

        # 3) 環境ルールに基づき無効手をマスク
        valid_actions = get_valid_actions(obs, env)
        probs = mask_probabilities(probs, valid_actions)

        # マスク後に合計が0なら打てる場所が無い
        if probs.sum() == 0:
            return 0

        # 4) 確率に従ってサンプリング
        action = np.random.choice(len(probs), p=probs)

        # 5) エピソードログに記録 (最初は報酬=0.0)
        self.episode_log.append(EpisodeStep(state=state_t, action=action, reward=0.0))
        return action

    def record_transition(self, s, a, r, s_next, done):
        """
        今回は get_action() 後に record_reward() で報酬を加えるため、ここでは何もしない
        """
        pass

    def record_reward(self, reward):
        """直近行動の報酬を上書き"""
        if self.episode_log:
            # dataclass のフィールドを更新
            self.episode_log[-1].reward = reward

    def _calc_returns(self) -> torch.Tensor:
        """エピソードログから割引報酬和を計算してテンソルを返す"""
        returns = []
        G = 0.0
        for step in reversed(self.episode_log):
            G = step.reward + self.gamma * G
            returns.append(G)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)

    def _optimize_model(self, states: torch.Tensor, actions: torch.Tensor, returns_t: torch.Tensor) -> None:
        """計算グラフを用いて方策ネットワークを更新する"""
        logits = self.model(states)
        log_probs = F.log_softmax(logits, dim=1)
        chosen_log_probs = log_probs[range(len(actions)), actions]
        loss = - (returns_t * chosen_log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def finish_episode(self):
        """
        REINFORCE (returns * log_prob)
        """
        if len(self.episode_log) == 0:
            return
        # 1) 割引報酬和の計算
        returns_t = self._calc_returns()

        # 2) 学習に使うテンソルへ変換
        states = torch.cat([step.state for step in self.episode_log], dim=0)
        actions = torch.tensor([step.action for step in self.episode_log], dtype=torch.long)

        # 3) 方策ネットワークを更新
        self._optimize_model(states, actions, returns_t)

        # 4) バッファ初期化
        self.episode_log = []

        # 5) エピソード毎に温度を下げる(任意)
        self.episode_count += 1
        self.update_temperature()

    def update_temperature(self):
        """
        エピソード終了ごとに温度を少しずつ下げる例。
        現在の self.temp に self.temp_decay を掛け、下限に達したら維持。
        """
        new_temp = self.temp * self.temp_decay
        self.temp = max(new_temp, self.min_temp)

    def save_model(self, path=MODEL_DIR / "policy_agent.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path=MODEL_DIR / "policy_agent.pth"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# ----------------------------------------------------
# QAgent (DQN)
# ----------------------------------------------------
class QNet(nn.Module):
    def __init__(self, board_size=9, hidden_size=128):
        super().__init__()
        input_dim = board_size * board_size
        output_dim = board_size * board_size
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        q = self.fc2(h)
        return q


class QAgent:
    def __init__(
        self,
        board_size=9,
        hidden_size=128,
        lr=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=5000,
        replay_capacity=10000,
        batch_size=64,
        update_frequency=10
    ):
        self.board_size = board_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_frequency = update_frequency

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_step = 0

        self.qnet = QNet(board_size, hidden_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        self.buffer = ReplayBuffer(replay_capacity)
        self.learn_step = 0

    def get_action(self, obs, env):
        valid_actions = get_valid_actions(obs, env)
        if random.random() < self.epsilon:
            if len(valid_actions) == 0:
                return 0
            action = random.choice(valid_actions)
        else:
            state_t = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.qnet(state_t).numpy().flatten()
            # 有効手以外は極端に低い値にする
            q_values = mask_q_values(q_values, valid_actions)
            action = int(np.argmax(q_values))

        # epsilon を線形に減衰
        self.epsilon_step += 1
        ratio = min(1.0, self.epsilon_step / self.epsilon_decay)
        self.epsilon = (1.0 - ratio) * self.epsilon + ratio * self.epsilon_end

        return action

    def record_transition(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)
        if len(self.buffer) >= self.batch_size and (self.learn_step % self.update_frequency == 0):
            self.train_on_batch()
        self.learn_step += 1

    def train_on_batch(self):
        s, a, r, s_next, d = self.buffer.sample(self.batch_size)

        # ReplayBuffer.sample() からは既に float32 の numpy 配列が返る
        # (batch, board, board) -> 学習では (batch, board^2) の形に変換
        states_np = s.reshape(self.batch_size, -1)
        next_states_np = s_next.reshape(self.batch_size, -1)

        states_t = torch.from_numpy(states_np)
        actions_t = torch.tensor(a, dtype=torch.long)
        rewards_t = torch.tensor(r, dtype=torch.float32)
        next_states_t = torch.from_numpy(next_states_np)
        dones_t = torch.tensor(d, dtype=torch.float32)

        q_values = self.qnet(states_t)
        q_a = q_values[range(self.batch_size), actions_t]

        with torch.no_grad():
            q_next = self.qnet(next_states_t)
            q_next_max = q_next.max(dim=1)[0]

        target = rewards_t + (1.0 - dones_t) * self.gamma * q_next_max
        loss = F.mse_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def finish_episode(self):
        """このエージェントでは未使用"""
        pass

    def save_model(self, path=MODEL_DIR / "q_agent.pth"):
        torch.save(self.qnet.state_dict(), path)

    def load_model(self, path=MODEL_DIR / "q_agent.pth"):
        self.qnet.load_state_dict(torch.load(path))
        self.qnet.eval()
