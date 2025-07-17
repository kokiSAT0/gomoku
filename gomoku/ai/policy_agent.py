"""PolicyAgent および関連クラス

方策勾配法で学習するエージェントの実装をまとめたモジュール。
コメントは日本語で詳細に記述し、読みやすさを重視する。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from pathlib import Path

from ..core.utils import get_valid_actions, mask_probabilities

# 学習済みモデルの保存先を定義
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)


@dataclass
class EpisodeStep:
    """PolicyAgent の 1 ステップ分の情報を保持するデータクラス"""

    state: torch.Tensor
    action: int
    reward: float


class PolicyNet(nn.Module):
    """盤面を入力として各マスの確率を出力する簡易ネットワーク"""

    def __init__(self, board_size: int = 9, hidden_size: int = 128) -> None:
        super().__init__()
        input_dim = board_size * board_size
        output_dim = board_size * board_size
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits


class PolicyAgent:
    """方策勾配で学習を行うエージェント"""

    def __init__(
        self,
        board_size: int = 9,
        hidden_size: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        temp: float = 2.0,
        device: str | None = None,
    ) -> None:
        self.board_size = board_size
        self.gamma = gamma

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = PolicyNet(board_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.episode_log: list[EpisodeStep] = []

        # 温度パラメータは徐々に下げて探索度を減らす
        self.temp = temp
        self.min_temp = 0.5
        self.temp_decay = 0.999

        self.episode_count = 0

    def get_action(self, obs: np.ndarray, env) -> int:
        """現在の方策に従って手を選択する"""
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

        action = int(np.random.choice(len(probs), p=probs))
        self.episode_log.append(EpisodeStep(state=state_t, action=action, reward=0.0))
        return action

    def record_transition(self, s, a, r, s_next, done) -> None:
        """外部から遷移を記録したい場合に使用 (現状未使用)"""
        pass

    def record_reward(self, reward: float) -> None:
        if self.episode_log:
            self.episode_log[-1].reward = reward

    def _calc_returns(self) -> torch.Tensor:
        """各ステップの累積報酬(割引報酬)を計算"""
        returns = []
        G = 0.0
        for step in reversed(self.episode_log):
            G = step.reward + self.gamma * G
            returns.append(G)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)

    def _optimize_model(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_t: torch.Tensor,
    ) -> None:
        """収集した軌跡を用いてパラメータを更新"""
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

    def finish_episode(self) -> None:
        """1 エピソード終了時に学習を実施"""
        if len(self.episode_log) == 0:
            return
        returns_t = self._calc_returns()
        states = torch.cat([step.state for step in self.episode_log], dim=0)
        actions = torch.tensor([step.action for step in self.episode_log], dtype=torch.long)
        self._optimize_model(states, actions, returns_t)
        self.episode_log = []
        self.episode_count += 1
        self.update_temperature()

    def update_temperature(self) -> None:
        """温度パラメータを少しずつ下げる"""
        new_temp = self.temp * self.temp_decay
        self.temp = max(new_temp, self.min_temp)

    def save_model(self, path: Path = MODEL_DIR / "policy_agent.pth") -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: Path = MODEL_DIR / "policy_agent.pth") -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
