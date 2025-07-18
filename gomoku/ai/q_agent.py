"""QAgent および関連クラス

DQN を利用した価値ベースのエージェントを実装するモジュール。
コメントは日本語で詳細に記述し、可読性を優先する。
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

from ..core.utils import get_valid_actions, mask_q_values, ReplayBuffer

# 学習済みモデルの保存先を定義
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)


class FCQNet(nn.Module):
    """盤面を一次元ベクトルとして処理する全結合型 Q ネットワーク"""

    def __init__(self, board_size: int = 9, hidden_size: int = 128) -> None:
        super().__init__()
        input_dim = board_size * board_size
        output_dim = board_size * board_size
        # 単純な二層の全結合ネットワーク
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        q = self.fc2(h)
        return q


class ConvQNet(nn.Module):
    """畳み込み層を用いた Q ネットワーク"""

    def __init__(self, board_size: int = 9, hidden_size: int = 64) -> None:
        super().__init__()
        self.board_size = board_size
        out_dim = board_size * board_size

        # 3x3 の畳み込み層を重ねて局所的な特徴を学習させる
        self.conv1 = nn.Conv2d(1, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)

        # 畳み込み後の特徴マップを全結合層へ入力
        self.fc = nn.Linear(hidden_size * board_size * board_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        h = x.view(b, 1, self.board_size, self.board_size)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(b, -1)
        q = self.fc(h)
        return q


# 後方互換性のため旧名称をエイリアスとして残す
QNet = FCQNet


class QAgent:
    """経験再生を用いた DQN エージェント"""

    def __init__(
        self,
        board_size: int = 9,
        hidden_size: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 5000,
        replay_capacity: int = 10000,
        batch_size: int = 64,
        update_frequency: int = 10,
        device: str | None = None,
        network_type: str = "fc",
    ) -> None:
        self.board_size = board_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_frequency = update_frequency

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_step = 0

        # デバイス指定が無ければ CUDA が利用可能かを確認
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ネットワークの種類を選択して初期化
        if network_type.lower() == "conv":
            self.qnet = ConvQNet(board_size, hidden_size).to(self.device)
        else:
            self.qnet = FCQNet(board_size, hidden_size).to(self.device)
        self.network_type = network_type
        # オプティマイザを設定
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        self.buffer = ReplayBuffer(replay_capacity)
        self.learn_step = 0

    def get_action(self, obs: np.ndarray, env) -> int:
        """ε-greedy 方策で行動を選択"""
        valid_actions = get_valid_actions(obs, env)
        if random.random() < self.epsilon:
            if len(valid_actions) == 0:
                return 0
            action = random.choice(valid_actions)
        else:
            # 盤面をテンソル化してネットワークへ入力
            state_t = (
                torch.tensor(obs.flatten(), dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                q_values = self.qnet(state_t).cpu().numpy().flatten()
            q_values = mask_q_values(q_values, valid_actions)
            action = int(np.argmax(q_values))

        self.epsilon_step += 1
        ratio = min(1.0, self.epsilon_step / self.epsilon_decay)
        self.epsilon = (1.0 - ratio) * self.epsilon + ratio * self.epsilon_end
        return action

    def record_transition(self, s, a, r, s_next, done) -> None:
        """経験をリプレイバッファへ格納し，必要に応じて学習"""
        self.buffer.push(s, a, r, s_next, done)
        if len(self.buffer) >= self.batch_size and (self.learn_step % self.update_frequency == 0):
            self.train_on_batch()
        self.learn_step += 1

    def train_on_batch(self) -> None:
        """バッファからサンプルしたミニバッチで DQN 更新を行う"""
        s, a, r, s_next, d = self.buffer.sample(self.batch_size)
        states_np = s.reshape(self.batch_size, -1)
        next_states_np = s_next.reshape(self.batch_size, -1)

        # NumPy 配列をテンソル化し GPU へ転送
        states_t = torch.from_numpy(states_np).to(self.device)
        actions_t = torch.tensor(a, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(r, dtype=torch.float32).to(self.device)
        next_states_t = torch.from_numpy(next_states_np).to(self.device)
        dones_t = torch.tensor(d, dtype=torch.float32).to(self.device)

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

    def finish_episode(self) -> None:
        """一括更新は行わないため空実装"""
        pass

    def save_model(self, path: Path = MODEL_DIR / "q_agent.pth") -> None:
        torch.save(self.qnet.state_dict(), path)

    def load_model(self, path: Path = MODEL_DIR / "q_agent.pth") -> None:
        # 保存されている重みを現在のデバイスへ読み込む
        state = torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(state)
        self.qnet.to(self.device)
        self.qnet.eval()
