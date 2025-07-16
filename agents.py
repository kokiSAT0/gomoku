# agents.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections

# ----------------------------------------------------
# 有効手(空マス)を列挙する関数 (ルール制限を省略した簡単版)
# ----------------------------------------------------
def get_valid_actions(obs, env):
    """
    obs: (board_size, board_size) numpy配列
    env: GomokuEnv (board_sizeなどを参照可能)
    戻り値: 空マスの action (0~board_size*board_size -1) 一覧
    """
    valid_actions = []
    board_size = obs.shape[0]
    for x in range(board_size):
        for y in range(board_size):
            # can_place_stone でチェック
            if env.can_place_stone(x, y):
                valid_actions.append(x * board_size + y)
    return valid_actions

def longest_chain_length(obs, x, y, player, directions):
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

def has_n_in_a_row(obs, x, y, player, directions, n):
    """n連以上が成立するかどうかを判定"""
    return longest_chain_length(obs, x, y, player, directions) >= n

# ----------------------------------------------------
# ランダムエージェント (学習しない)
# ----------------------------------------------------
class RandomAgent:
    def __init__(self):
        pass

    def get_action(self, obs, env):
        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0
        return random.choice(valid_actions)

    def record_transition(self, s, a, r, s_next, done):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path="random_agent.pth"):
        pass

    def load_model(self, path="random_agent.pth"):
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
        board_size = obs.shape[0]
        current_player = env.current_player
        opponent = 1 if current_player == 2 else 2

        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0

        # 1) 自分が置けば5連になるか？
        move = self.find_winning_move(obs, valid_actions, current_player)
        if move is not None:
            return move

        # 2) 相手が次手で5連できるならブロック
        move = self.find_winning_move(obs, valid_actions, opponent)
        if move is not None:
            return move

        # 3) 上記がなければランダム
        return random.choice(valid_actions)

    def find_winning_move(self, obs, valid_actions, player):
        board_size = obs.shape[0]
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for a in valid_actions:
            x = a // board_size
            y = a % board_size
            # 仮置き
            obs[x,y] = player
            if self.check_5_in_a_row(obs, x, y, player, directions):
                obs[x,y] = 0
                return a
            obs[x,y] = 0
        return None

    def check_5_in_a_row(self, obs, x, y, player, directions):
        """5連になるかどうかをヘルパーで判定"""
        return has_n_in_a_row(obs, x, y, player, directions, 5)

    def record_transition(self, s, a, r, s_next, done):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path="immediate_win_block_agent.pth"):
        pass

    def load_model(self, path="immediate_win_block_agent.pth"):
        pass


class FourThreePriorityAgent:
    """
    4連/3連を優先し、ブロックもする簡易ヒューリスティック
    """
    def __init__(self):
        pass

    def get_action(self, obs, env):
        board_size = obs.shape[0]
        current_player = env.current_player
        opponent = 1 if current_player == 2 else 2

        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0

        # 1) 自分の4連が完成する手
        a = self.find_n_chain(obs, valid_actions, current_player, 4)
        if a is not None:
            return a

        # 2) 相手の4連ブロック
        a = self.find_n_chain(obs, valid_actions, opponent, 4)
        if a is not None:
            return a

        # 3) 自分の3連が完成する手
        a = self.find_n_chain(obs, valid_actions, current_player, 3)
        if a is not None:
            return a

        # 4) 相手の3連ブロック
        a = self.find_n_chain(obs, valid_actions, opponent, 3)
        if a is not None:
            return a

        # 5) ランダム
        return random.choice(valid_actions)

    def find_n_chain(self, obs, valid_actions, player, n):
        board_size = obs.shape[0]
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for a in valid_actions:
            x = a // board_size
            y = a % board_size
            obs[x,y] = player
            if self.check_n_chain(obs, x, y, player, directions, n):
                obs[x,y] = 0
                return a
            obs[x,y] = 0
        return None

    def check_n_chain(self, obs, x, y, player, directions, n):
        """n連以上になるかを判定"""
        return has_n_in_a_row(obs, x, y, player, directions, n)

    def record_transition(self, s, a, r, s_next, done):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path="four_three_priority_agent.pth"):
        pass

    def load_model(self, path="four_three_priority_agent.pth"):
        pass


class LongestChainAgent:
    """
    置いたときに最長の連ができる手を優先する簡易ヒューリスティック
    """
    def __init__(self):
        pass

    def get_action(self, obs, env):
        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0

        current_player = env.current_player
        best_score = -1
        best_actions = []
        directions = [(1,0),(0,1),(1,1),(1,-1)]

        for a in valid_actions:
            x = a // env.board_size
            y = a % env.board_size
            # 仮置き
            obs[x,y] = current_player
            score = self.get_longest_chain(obs, x, y, current_player, directions)
            obs[x,y] = 0

            if score > best_score:
                best_score = score
                best_actions = [a]
            elif score == best_score:
                best_actions.append(a)

        return random.choice(best_actions)

    def get_longest_chain(self, obs, x, y, player, directions):
        """最長連結長を返す (ヘルパー関数のラッパー)"""
        return longest_chain_length(obs, x, y, player, directions)

    def record_transition(self, s, a, r, s_next, done):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path="longest_chain_agent.pth"):
        pass

    def load_model(self, path="longest_chain_agent.pth"):
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
        self.episode_log = []  # [ [state_t, action, reward], ... ]

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
            logits = self.model(state_t)  # shape: (1, board_size*board_size)
            # ここでソフトマックス温度を適用
            scaled_logits = logits / self.temp
            probs = F.softmax(scaled_logits, dim=1).cpu().numpy().flatten()

        # 3) 環境ルールに基づいて invalid(無効)手を除外
        valid_actions = get_valid_actions(obs, env)
        mask = np.zeros_like(probs, dtype=bool)
        for a in valid_actions:
            mask[a] = True
        probs[~mask] = 0.0

        if probs.sum() == 0:
            # 有効手が無い → 強制的に action=0 など
            return 0

        # 4) 確率を再正規化してサンプリング
        probs /= probs.sum()
        action = np.random.choice(len(probs), p=probs)

        # 5) エピソードログに記録 (最初は報酬=0.0)
        self.episode_log.append([state_t, action, 0.0])
        return action

    def record_transition(self, s, a, r, s_next, done):
        """
        今回は get_action() 後に record_reward() で報酬を加えるため、ここでは何もしない
        """
        pass

    def record_reward(self, reward):
        """直近行動の報酬を上書き"""
        if self.episode_log:
            self.episode_log[-1][2] = reward

    def finish_episode(self):
        """
        REINFORCE (returns * log_prob)
        """
        if len(self.episode_log) == 0:
            return

        # 1) 割引報酬和を後ろから計算
        returns = []
        G = 0.0
        for i in reversed(range(len(self.episode_log))):
            G = self.episode_log[i][2] + self.gamma * G
            returns.insert(0, G)

        # 2) データ整形
        states = torch.cat([x[0] for x in self.episode_log], dim=0)  # (T, board_size^2)
        actions = torch.tensor([x[1] for x in self.episode_log], dtype=torch.long)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # 3) 順伝播して log_prob を計算
        logits = self.model(states)
        log_probs = F.log_softmax(logits, dim=1)    # (T, board_size^2)
        chosen_log_probs = log_probs[range(len(actions)), actions]

        # 4) REINFORCE損失 = - (returns * log_probs).mean()
        loss = - (returns_t * chosen_log_probs).mean()

        # 5) 学習更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6) バッファ初期化
        self.episode_log = []

        # 7) エピソード毎に温度を下げる(任意)
        self.episode_count += 1
        self.update_temperature()

    def update_temperature(self):
        """
        エピソード終了ごとに温度を少しずつ下げる例。
        現在の self.temp に self.temp_decay を掛け、下限に達したら維持。
        """
        new_temp = self.temp * self.temp_decay
        self.temp = max(new_temp, self.min_temp)

    def save_model(self, path="policy_agent.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="policy_agent.pth"):
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

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return s, a, r, s_next, d

    def __len__(self):
        return len(self.buffer)

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
            # 有効手以外は -∞
            mask = np.ones_like(q_values, dtype=bool)
            for a in valid_actions:
                mask[a] = False
            q_values[mask] = -1e9
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

        # 修正: まとめて numpy.array に変換してから Tensor 化
        states_np = np.array([arr.flatten() for arr in s], dtype=np.float32)
        next_states_np = np.array([arr.flatten() for arr in s_next], dtype=np.float32)

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
        pass

    def save_model(self, path="q_agent.pth"):
        torch.save(self.qnet.state_dict(), path)

    def load_model(self, path="q_agent.pth"):
        self.qnet.load_state_dict(torch.load(path))
        self.qnet.eval()
