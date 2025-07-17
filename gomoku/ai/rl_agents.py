"""強化学習ベースのエージェント群"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

from ..core.utils import (
    get_valid_actions,
    mask_probabilities,
    mask_q_values,
    ReplayBuffer,
)

# 学習済みモデルの保存先
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)


@dataclass
class EpisodeStep:
    """PolicyAgent の 1 ステップ分の情報"""

    state: torch.Tensor
    action: int
    reward: float


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
    def __init__(
        self,
        board_size=9,
        hidden_size=128,
        lr=1e-3,
        gamma=0.99,
        temp=2.0,
        device=None,
    ):
        self.board_size = board_size
        self.gamma = gamma

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = PolicyNet(board_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.episode_log: list[EpisodeStep] = []

        self.temp = temp
        self.min_temp = 0.5
        self.temp_decay = 0.999

        self.episode_count = 0

    def get_action(self, obs, env):
        state_t = (
            torch.tensor(obs.flatten(), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.model(state_t)
            scaled_logits = logits / self.temp
            probs = F.softmax(scaled_logits, dim=1).cpu().numpy().flatten()

        valid_actions = get_valid_actions(obs, env)
        probs = mask_probabilities(probs, valid_actions)

        if probs.sum() == 0:
            return 0

        action = np.random.choice(len(probs), p=probs)
        self.episode_log.append(EpisodeStep(state=state_t, action=action, reward=0.0))
        return action

    def record_transition(self, s, a, r, s_next, done):
        pass

    def record_reward(self, reward):
        if self.episode_log:
            self.episode_log[-1].reward = reward

    def _calc_returns(self) -> torch.Tensor:
        returns = []
        G = 0.0
        for step in reversed(self.episode_log):
            G = step.reward + self.gamma * G
            returns.append(G)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)

    def _optimize_model(self, states: torch.Tensor, actions: torch.Tensor, returns_t: torch.Tensor) -> None:
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns_t = returns_t.to(self.device)

        logits = self.model(states)
        log_probs = F.log_softmax(logits, dim=1)
        chosen_log_probs = log_probs[range(len(actions)), actions]
        loss = -(returns_t * chosen_log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def finish_episode(self):
        if len(self.episode_log) == 0:
            return
        returns_t = self._calc_returns()
        states = torch.cat([step.state for step in self.episode_log], dim=0)
        actions = torch.tensor([step.action for step in self.episode_log], dtype=torch.long)
        self._optimize_model(states, actions, returns_t)
        self.episode_log = []
        self.episode_count += 1
        self.update_temperature()

    def update_temperature(self):
        new_temp = self.temp * self.temp_decay
        self.temp = max(new_temp, self.min_temp)

    def save_model(self, path=MODEL_DIR / "policy_agent.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path=MODEL_DIR / "policy_agent.pth"):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()


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
        update_frequency=10,
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
            q_values = mask_q_values(q_values, valid_actions)
            action = int(np.argmax(q_values))

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
        pass

    def save_model(self, path=MODEL_DIR / "q_agent.pth"):
        torch.save(self.qnet.state_dict(), path)

    def load_model(self, path=MODEL_DIR / "q_agent.pth"):
        self.qnet.load_state_dict(torch.load(path))
        self.qnet.eval()
