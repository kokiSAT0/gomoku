import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from utils import moving_average, opponent_player

# play_with_pygame の描画機能を利用するためのインポート
from play_with_pygame import play_game

# 学習済みモデルを保存するディレクトリ
MODEL_DIR = Path(__file__).resolve().parent / "models"

########################################################
# 1) GomokuEnv (五目並べ環境)
#    - 連数を数える関数
#    - 五目並べの基本ルール実装
#    - 強化学習向けstep()で報酬を付与
########################################################

def count_chains_open_ends(board: np.ndarray, player: int):
    """
    board上で指定playerの
    - ちょうど2連 / 3連 / 4連
      それぞれについて
       (open2: 両端空, open1: 片端のみ空) の本数を数える。
    返り値は辞書形式で:
    {
      2: {'open2': 個数, 'open1': 個数},
      3: {'open2': 個数, 'open1': 個数},
      4: {'open2': 個数, 'open1': 個数}
    }
    のようにまとめて返す。
    """
    board_size = board.shape[0]
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    result = {
        2: {'open2': 0, 'open1': 0},
        3: {'open2': 0, 'open1': 0},
        4: {'open2': 0, 'open1': 0},
    }

    for x in range(board_size):
        for y in range(board_size):
            if board[x, y] != player:
                continue

            for dx, dy in directions:
                # 逆方向に同じプレイヤーの石があるなら「連の開始点」ではないのでスキップ
                prev_x, prev_y = x - dx, y - dy
                if 0 <= prev_x < board_size and 0 <= prev_y < board_size:
                    if board[prev_x, prev_y] == player:
                        continue

                # (x, y)から(dx, dy)方向に連がいくつ続いているか
                length = 1
                cx, cy = x + dx, y + dy
                while 0 <= cx < board_size and 0 <= cy < board_size and board[cx, cy] == player:
                    length += 1
                    cx += dx
                    cy += dy

                # lengthが2,3,4のときのみカウント
                if length in [2, 3, 4]:
                    open_ends = 0
                    # 連の前端
                    if 0 <= prev_x < board_size and 0 <= prev_y < board_size:
                        if board[prev_x, prev_y] == 0:
                            open_ends += 1
                    # 連の後端(cx, cy は「連が切れた場所」なので空かどうかチェック)
                    if 0 <= cx < board_size and 0 <= cy < board_size:
                        if board[cx, cy] == 0:
                            open_ends += 1

                    if open_ends == 2:
                        result[length]['open2'] += 1
                    elif open_ends == 1:
                        result[length]['open1'] += 1

    return result


class Gomoku:
    """
    五目並べのゲームロジック。
    board_size x board_size の盤面を扱い、
      - 盤面の初期化
      - 石を置けるかどうかのチェック
      - 石を置いたあとの勝敗判定
    などを行う。
    """
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 先手=1(黒), 後手=2(白)

    def reset(self):
        """盤面をリセットして先手を黒番(1)にする。"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1

    def is_valid_move(self, x, y):
        """(x, y)が盤内であり、かつ空マスならTrue"""
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False
        return (self.board[x, y] == 0)

    def place_stone(self, x, y):
        """
        (x, y)に現在のプレイヤーの石を置き、
        手番を相手プレイヤーに交替。
        """
        self.board[x, y] = self.current_player
        # 石を置いたら手番を交替
        self.current_player = opponent_player(self.current_player)

    def check_winner(self):
        """
        盤面を見て、勝者がいるかどうかを判定する。
        - 1(黒)勝ちなら 1
        - 2(白)勝ちなら 2
        - 引き分けなら -1
        - まだ決着ついていなければ 0
        """
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for x in range(self.board_size):
            for y in range(self.board_size):
                player = self.board[x, y]
                if player == 0:
                    continue
                for dx, dy in directions:
                    if self._count_stones(x, y, dx, dy, player) >= 5:
                        return player
        # 盤が全て埋まっていたら引き分け
        if np.all(self.board != 0):
            return -1
        return 0

    def _count_stones(self, x, y, dx, dy, player):
        """
        (x,y)から(dx,dy)方向に連続している同色(player)の石を数える。
        """
        count = 0
        cx, cy = x, y
        while 0 <= cx < self.board_size and 0 <= cy < self.board_size:
            if self.board[cx, cy] == player:
                count += 1
                cx += dx
                cy += dy
            else:
                break
        return count


class GomokuEnv:
    """
    五目並べを「強化学習用の環境」としてラップし、
    中間報酬(連を作る・相手の連をブロックする)を追加したバージョン。
    
    使い方:
      env = GomokuEnv()
      obs = env.reset()
      obs, reward, done, info = env.step(action)
      ...
    """
    def __init__(
        self,
        board_size=9,
        force_center_first_move=True,
        adjacency_range=1,
        invalid_move_penalty=-1.0,
        # 以下、中間報酬の設定値(自由に調整可能)
        reward_chain_2_open2=0.01,
        reward_chain_3_open2=0.2,
        reward_chain_4_open2=0.5,
        reward_chain_2_open1=0,
        reward_chain_3_open1=0.05,
        reward_chain_4_open1=0.4,
        reward_block_2_open2=0.05,
        reward_block_3_open2=0.3,
        reward_block_4_open2=0,
        reward_block_2_open1=0.0,
        reward_block_3_open1=0.05,
        reward_block_4_open1=0.5
    ):
        self.board_size = board_size
        self.game = Gomoku(board_size)
        self.done = False
        self.winner = 0
        self.turn_count = 0

        # 初手を中央に強制するか
        self.force_center_first_move = force_center_first_move
        # 2手目以降、既存の石から adjacency_range 以内かどうか (コメントアウトで無効化可能)
        self.adjacency_range = adjacency_range
        # 無効手を打ったときのペナルティ
        self.invalid_move_penalty = invalid_move_penalty

        # 中間報酬に関するテーブル (自分が連を作ったとき)
        self.r_chain = {
            2: {'open2': reward_chain_2_open2, 'open1': reward_chain_2_open1},
            3: {'open2': reward_chain_3_open2, 'open1': reward_chain_3_open1},
            4: {'open2': reward_chain_4_open2, 'open1': reward_chain_4_open1},
        }
        # 中間報酬に関するテーブル (相手の連をブロックしたとき)
        self.r_block = {
            2: {'open2': reward_block_2_open2, 'open1': reward_block_2_open1},
            3: {'open2': reward_block_3_open2, 'open1': reward_block_3_open1},
            4: {'open2': reward_block_4_open2, 'open1': reward_block_4_open1},
        }

    def reset(self):
        self.game.reset()
        self.done = False
        self.winner = 0
        self.turn_count = 0
        return self._get_observation()

    def action_to_coord(self, action: int) -> tuple[int, int]:
        """action 番号を ``(x, y)`` へ変換するヘルパー"""
        x = action // self.board_size
        y = action % self.board_size
        return x, y

    def coord_to_action(self, x: int, y: int) -> int:
        """``(x, y)`` 座標を action 番号に変換"""
        return x * self.board_size + y
    
    def can_place_stone(self, x, y):
        """
        自分の環境ルールに照らして(x,y)が有効手かどうかをチェック。
        - 1) 初手は中央強制の場合、(turn_count == 0) なら (x==center, y==center) か？
        - 2) adjacency_range 制限を適用するなら適用 (コメントアウト可能)
        - 3) 盤外 or 既に石ありでないか？
        
        全て満たせばTrue, そうでなければFalse。
        """
        # 1) 初手は中央のみ
        if self.turn_count == 0 and self.force_center_first_move:
            center = self.board_size // 2
            if not (x == center and y == center):
                return False
        
        # 2) adjacency_rangeチェック(必要ならコメントアウト解除)
        # if (self.turn_count >= 1) and (self.adjacency_range is not None) and (self.adjacency_range > 0):
        #     if not self._is_adjacent_to_stone(x, y, self.adjacency_range):
        #         return False
        
        # 3) 盤外 or 既に石あり
        if not self.game.is_valid_move(x, y):
            return False

        return True

    def step(self, action):
        """
        ``action`` は ``0`` から ``board_size**2 - 1`` の整数。
        ``action_to_coord()`` を使って座標 ``(x, y)`` に変換し着手する。
        戻り値: ``(obs, reward, done, info)``
        """
        if self.done:
            # 既に終わっている場合、報酬0で終了状態を返す
            return self._get_observation(), 0.0, True, {"winner": self.winner}

        # 現在のプレイヤーと相手プレイヤーを取得
        current_player = self.game.current_player
        # ユーティリティ関数で相手プレイヤーを取得
        opponent = opponent_player(current_player)

        # 着手前の連数を計測
        before_self = count_chains_open_ends(self.game.board, current_player)
        before_opp = count_chains_open_ends(self.game.board, opponent)

        # 座標に変換
        x, y = self.action_to_coord(action)

        # 無効手チェック
        if not self.can_place_stone(x, y):
            self.done = True
            self.winner = 0  # 勝者なし
            return self._get_observation(), self.invalid_move_penalty, True, {
                "winner": 0, "reason": "invalid_move"
            }

        # 有効手なので石を置く
        self.game.place_stone(x, y)
        self.turn_count += 1

        # 決着判定
        winner = self.game.check_winner()
        reward_final = 0.0
        if winner != 0:
            # 勝ち(1 or 2) or 引き分け(-1)
            self.done = True
            self.winner = winner
            if winner == 1:
                # 黒勝ち → 打ったのが黒なら +1.0, 白なら -1.0
                reward_final = 1.0 if current_player == 1 else -1.0
            elif winner == 2:
                # 白勝ち → 打ったのが白なら +1.0, 黒なら -1.0
                reward_final = 1.0 if current_player == 2 else -1.0
            else:
                # 引き分け(-1) の場合
                reward_final = -1
        else:
            self.done = False

        # 中間報酬(連の増減)
        reward_intermediate = 0.0
        if not self.done:
            after_self = count_chains_open_ends(self.game.board, current_player)
            after_opp = count_chains_open_ends(self.game.board, opponent)

            # 自分の 2連/3連/4連 が増えたら加点
            for length in [2, 3, 4]:
                for open_type in ['open2', 'open1']:
                    diff = after_self[length][open_type] - before_self[length][open_type]
                    if diff > 0:
                        reward_intermediate += diff * self.r_chain[length][open_type]

            # 相手の 2連/3連/4連 が減ったら加点 (ブロック報酬)
            for length in [2, 3, 4]:
                for open_type in ['open2', 'open1']:
                    diff = before_opp[length][open_type] - after_opp[length][open_type]
                    if diff > 0:
                        reward_intermediate += diff * self.r_block[length][open_type]

        total_reward = reward_final + reward_intermediate

        return self._get_observation(), total_reward, self.done, {"winner": self.winner}

    def render(self):
        """人間向けのテキスト表示"""
        print("   ", end="")
        for y in range(self.board_size):
            print(f"{y:2d}", end=" ")
        print()
        for x in range(self.board_size):
            print(f"{x:2d} ", end="")
            for y in range(self.board_size):
                if self.game.board[x, y] == 0:
                    print(".", end="  ")
                elif self.game.board[x, y] == 1:
                    print("X", end="  ")
                else:
                    print("O", end="  ")
            print()

    def _get_observation(self):
        """現在の盤面をコピーして返す。"""
        return self.game.board.copy()

    @property
    def current_player(self):
        """現在手番のプレイヤーIDを返す(1 or 2)"""
        return self.game.current_player

    def _is_adjacent_to_stone(self, x, y, rng):
        """
        盤上の任意の石とのチェビシェフ距離が rng 以内なら True。
        rng=1 であれば、(x±1,y±1) 含む周囲1マス以内に石があれば True。
        """
        board = self.game.board
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] != 0:
                    if abs(i - x) <= rng and abs(j - y) <= rng:
                        return True
        return False


########################################################
# 2) 各種エージェント
########################################################

def get_valid_actions(obs, env):
    """
    obs: (board_size, board_size) numpy配列
    env: GomokuEnv (board_sizeなどを参照可能)
    戻り値: 空マスで、かつ envルールで打てる action (0~board_size*board_size -1) 一覧
    """
    valid_actions = []
    board_size = obs.shape[0]
    for x in range(board_size):
        for y in range(board_size):
            if env.can_place_stone(x, y):
                valid_actions.append(env.coord_to_action(x, y))
    return valid_actions

# ----------------------------------------------------
# 方策分布/Q値のマスク処理用ヘルパー
# ----------------------------------------------------

def mask_probabilities(probs: np.ndarray, valid_actions: list[int]) -> np.ndarray:
    """有効手のみ残して正規化した確率配列を返す"""
    masked = np.zeros_like(probs)
    for a in valid_actions:
        masked[a] = probs[a]
    total = masked.sum()
    if total > 0.0:
        masked /= total
    return masked


def mask_q_values(q_values: np.ndarray, valid_actions: list[int], invalid_value: float = -1e9) -> np.ndarray:
    """無効手のQ値を ``invalid_value`` で置き換える"""
    masked = q_values.copy()
    mask = np.ones_like(masked, dtype=bool)
    for a in valid_actions:
        mask[a] = False
    masked[mask] = invalid_value
    return masked


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

    def record_reward(self, r):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path=MODEL_DIR / "random_agent.pth"):
        pass

    def load_model(self, path=MODEL_DIR / "random_agent.pth"):
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
        # 相手プレイヤーIDを取得
        opponent = opponent_player(current_player)

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
        board_size = obs.shape[0]
        for dx, dy in directions:
            count = 1
            # 正方向
            cx, cy = x+dx, y+dy
            while 0 <= cx<board_size and 0 <= cy<board_size and obs[cx,cy] == player:
                count += 1
                cx += dx
                cy += dy
            # 逆方向
            cx, cy = x-dx, y-dy
            while 0 <= cx<board_size and 0 <= cy<board_size and obs[cx,cy] == player:
                count += 1
                cx -= dx
                cy -= dy

            if count >= 5:
                return True
        return False

    def record_transition(self, s, a, r, s_next, done):
        pass

    def record_reward(self, r):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path=MODEL_DIR / "immediate_win_block_agent.pth"):
        pass

    def load_model(self, path=MODEL_DIR / "immediate_win_block_agent.pth"):
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
        # 相手プレイヤーIDを取得
        opponent = opponent_player(current_player)

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
        board_size = obs.shape[0]
        for dx, dy in directions:
            count = 1
            # 正
            cx, cy = x+dx, y+dy
            while 0 <= cx<board_size and 0 <= cy<board_size and obs[cx,cy] == player:
                count += 1
                cx += dx
                cy += dy
            # 逆
            cx, cy = x-dx, y-dy
            while 0 <= cx<board_size and 0 <= cy<board_size and obs[cx,cy] == player:
                count += 1
                cx -= dx
                cy -= dy
            if count >= n:
                return True
        return False

    def record_transition(self, s, a, r, s_next, done):
        pass

    def record_reward(self, r):
        pass

    def finish_episode(self):
        pass

    def save_model(self, path=MODEL_DIR / "four_three_priority_agent.pth"):
        pass

    def load_model(self, path=MODEL_DIR / "four_three_priority_agent.pth"):
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
            # action 番号を座標へ変換
            x, y = env.action_to_coord(a)
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
        max_len = 1
        for dx, dy in directions:
            count = 1
            # 正
            cx, cy = x+dx, y+dy
            while 0 <= cx<obs.shape[0] and 0 <= cy<obs.shape[1] and obs[cx,cy] == player:
                count += 1
                cx += dx
                cy += dy
            # 逆
            cx, cy = x-dx, y-dy
            while 0 <= cx<obs.shape[0] and 0 <= cy<obs.shape[1] and obs[cx,cy] == player:
                count += 1
                cx -= dx
                cy -= dy
            max_len = max(max_len, count)
        return max_len

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


# ----------------------------------------------------
# PolicyAgent (REINFORCE + エントロピー正則化 + 簡易ベースライン)
# ----------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, board_size=9, hidden_size=128):
        super().__init__()
        input_dim = board_size * board_size
        output_dim = board_size * board_size
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x shape: (batch_size, board_size*board_size)
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
        min_temp=0.5,
        temp_decay=0.999,
        entropy_coef=0.01,
    ):
        """
        - temp: ソフトマックス温度 (初期値)
        - entropy_coef: エントロピー正則化係数
        """
        self.board_size = board_size
        self.gamma = gamma
        self.temp = temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        self.entropy_coef = entropy_coef

        self.model = PolicyNet(board_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # メモリ: (state_t, action, reward, logits) を保存する
        self.episode_log = []

        # リターンの移動平均(単純に使うだけの簡易ベースライン)
        self.return_running_mean = 0.0
        self.alpha_baseline = 0.01  # 移動平均の更新率

        self.episode_count = 0

    def get_action(self, obs, env):
        """
        obs: (board_size, board_size) の numpy配列
        env: GomokuEnv
        戻り値: action(int)
        """
        # 1) 盤面をflattenしてテンソル化
        state_t = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)

        # 2) ネットワーク推論 (logits)  ← 勾配を追跡したいので no_grad() は付けない
        logits = self.model(state_t)  # shape: (1, board_size*board_size)

        # ソフトマックス温度を適用
        scaled_logits = logits / self.temp
        probs_tensor = F.softmax(scaled_logits, dim=1)  # 勾配追跡あり

        # 3) 有効手以外をマスクするために一度 NumPyに変換して操作したい
        #    → .detach() してから .cpu().numpy() を呼び出す
        probs_np = probs_tensor.detach().cpu().numpy().flatten()

        valid_actions = get_valid_actions(obs, env)
        probs_np = mask_probabilities(probs_np, valid_actions)

        if probs_np.sum() == 0.0:
            return 0  # 有効手が無い場合のfallback

        action = np.random.choice(len(probs_np), p=probs_np)

        # 4) エピソードログに記録 (最初は報酬=0.0)
        #    学習フェーズで log_probs を計算するため logits をそのまま保存する
        self.episode_log.append([state_t, action, 0.0, logits])

        return action

    def record_transition(self, s, a, r, s_next, done):
        """
        今回は get_action() の中でエピソードログに state,action を保存し、
        報酬は別メソッドで更新するので、ここでは何もしない想定。
        """
        pass

    def record_reward(self, reward):
        """直近行動の報酬を上書き"""
        if self.episode_log:
            self.episode_log[-1][2] = reward

    def finish_episode(self):
        """
        REINFORCE (returns * log_prob) に
        - エントロピー正則化
        - 簡易ベースライン
        を加えて学習を行う。
        """
        if len(self.episode_log) == 0:
            return

        # 1) 割引報酬和を後ろから計算
        returns = []
        G = 0.0
        for i in reversed(range(len(self.episode_log))):
            G = self.episode_log[i][2] + self.gamma * G
            returns.insert(0, G)

        # 2) リターンの移動平均を更新(簡易ベースライン用)
        episode_return = returns[0]  # 最初の行動が終局時点までの割引和
        self.return_running_mean = ((1.0 - self.alpha_baseline)*self.return_running_mean
                                    + self.alpha_baseline*episode_return)

        # 3) データ整形
        states = torch.cat([x[0] for x in self.episode_log], dim=0)  # (T, board_size^2)
        actions = [x[1] for x in self.episode_log]
        returns_t = torch.tensor(returns, dtype=torch.float32)
        logits_list = [x[3] for x in self.episode_log]
        logits_cat = torch.cat(logits_list, dim=0)  # shape: (T, board_size^2)

        # 4) 順伝播して log_prob, entropy を計算 (エピソード分まとめて一気に)
        log_probs = F.log_softmax(logits_cat, dim=1)  # (T, board_size^2)
        probs = F.softmax(logits_cat, dim=1)         # (T, board_size^2)
        chosen_log_probs = log_probs[range(len(actions)), actions]
        # エントロピー(全行動に対して -p*log(p) の和)
        entropy = -(log_probs * probs).sum(dim=1).mean()

        # 5) REINFORCE損失 (ベースラインは return_running_mean を引く)
        baseline = self.return_running_mean
        advantage = returns_t - baseline  # 簡易ベースラインとして引く
        loss_policy = - (advantage * chosen_log_probs).mean()
        loss_entropy = - self.entropy_coef * entropy  # エントロピーを最大化したいので "-" 付与

        loss = loss_policy + loss_entropy

        # 6) 学習更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 7) バッファ初期化
        self.episode_log = []

        # 8) エピソード毎に温度を下げる(任意)
        self.episode_count += 1
        self._update_temperature()

    def _update_temperature(self):
        new_temp = self.temp * self.temp_decay
        self.temp = max(new_temp, self.min_temp)

    def save_model(self, path=MODEL_DIR / "policy_agent.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path=MODEL_DIR / "policy_agent.pth"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


# ----------------------------------------------------
# QAgent (DQN + ターゲットネットワーク)
# ----------------------------------------------------

class QNet(nn.Module):
    def __init__(self, board_size=9, hidden_size=128):
        super().__init__()
        input_dim = board_size * board_size
        output_dim = board_size * board_size
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x shape: (batch_size, board_size^2)
        h = F.relu(self.fc1(x))
        q = self.fc2(h)
        return q


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        """ランダムに ``batch_size`` 件を取り出し数値配列で返す"""

        # リプレイバッファからランダムに抽出
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)

        # dtype を明示しつつ stack して扱いやすくする
        states = np.stack(s).astype(np.float32)
        next_states = np.stack(s_next).astype(np.float32)
        actions = np.array(a, dtype=np.int64)
        rewards = np.array(r, dtype=np.float32)
        dones = np.array(d, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def hard_update(dst_net, src_net):
    """ターゲットネットワークを同じ重みにする(ハードコピー)"""
    dst_net.load_state_dict(src_net.state_dict())


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
        target_update_frequency=200
    ):
        self.board_size = board_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_step = 0

        self.qnet = QNet(board_size, hidden_size)
        self.target_qnet = QNet(board_size, hidden_size)
        hard_update(self.target_qnet, self.qnet)  # 初期は同じにしておく

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.buffer = ReplayBuffer(replay_capacity)
        self.learn_step = 0

    def get_action(self, obs, env):
        valid_actions = get_valid_actions(obs, env)
        if not valid_actions:
            return 0

        # ε-greedy で行動
        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            state_t = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.qnet(state_t).numpy().flatten()
            # 無効手は非常に低い値に置き換える
            q_values = mask_q_values(q_values, valid_actions)
            action = int(np.argmax(q_values))

        # epsilon を線形に減衰
        self.epsilon_step += 1
        ratio = min(1.0, self.epsilon_step / self.epsilon_decay)
        self.epsilon = (1.0 - ratio) * self.epsilon + ratio * self.epsilon_end

        return action

    def record_transition(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)
        # 学習タイミング
        if len(self.buffer) >= self.batch_size and (self.learn_step % self.update_frequency == 0):
            self.train_on_batch()
        self.learn_step += 1

        # ターゲットネットワークの更新
        if self.learn_step % self.target_update_frequency == 0:
            hard_update(self.target_qnet, self.qnet)

    def train_on_batch(self):
        s, a, r, s_next, d = self.buffer.sample(self.batch_size)

        # sample() で float32 の配列として受け取れるので reshape のみ行う
        states_np = s.reshape(self.batch_size, -1)
        next_states_np = s_next.reshape(self.batch_size, -1)

        states_t = torch.from_numpy(states_np)
        actions_t = torch.tensor(a, dtype=torch.long)
        rewards_t = torch.tensor(r, dtype=torch.float32)
        next_states_t = torch.from_numpy(next_states_np)
        dones_t = torch.tensor(d, dtype=torch.float32)

        q_values = self.qnet(states_t)           # (batch_size, board_size^2)
        q_a = q_values[range(self.batch_size), actions_t]

        with torch.no_grad():
            q_next = self.target_qnet(next_states_t)
            q_next_max = q_next.max(dim=1)[0]

        target = rewards_t + (1.0 - dones_t) * self.gamma * q_next_max
        loss = F.mse_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def record_reward(self, r):
        # PolicyAgent とは違い、DQN では使わないので空
        pass

    def finish_episode(self):
        pass

    def save_model(self, path=MODEL_DIR / "q_agent.pth"):
        torch.save(self.qnet.state_dict(), path)

    def load_model(self, path=MODEL_DIR / "q_agent.pth"):
        self.qnet.load_state_dict(torch.load(path))
        self.qnet.eval()
        hard_update(self.target_qnet, self.qnet)


########################################################
# 3) 学習ループ関連
########################################################

def train_agents(
    env,
    black_agent,
    white_agent,
    episodes=1000,
):
    """
    黒番=black_agent, 白番=white_agent で自己対戦させる汎用学習ループ。
    戻り値: (all_rewards_black, all_rewards_white, winners, turns_list)
    """
    all_rewards_black = []
    all_rewards_white = []
    winners = []
    turns_list = []

    for epi in tqdm(range(episodes), desc="train_agents"):
        obs = env.reset()
        done = False
        black_episode_reward = 0.0
        white_episode_reward = 0.0

        while not done:
            current_player = env.current_player
            s = obs.copy()

            if current_player == 1:
                action = black_agent.get_action(obs, env)
            else:
                action = white_agent.get_action(obs, env)

            next_obs, reward, done, info = env.step(action)

            # 報酬を各エージェントに記録
            if current_player == 1:
                # 黒番
                black_agent.record_reward(reward)
                black_agent.record_transition(s, action, reward, next_obs, done)
                black_episode_reward += reward
            else:
                # 白番
                white_agent.record_reward(reward)
                white_agent.record_transition(s, action, reward, next_obs, done)
                white_episode_reward += reward

            obs = next_obs

        # エピソード終了
        black_agent.finish_episode()
        white_agent.finish_episode()

        w = info["winner"]  # 1(黒勝),2(白勝),-1(引),0(無効手)
        winners.append(w)
        turns_list.append(env.turn_count)
        all_rewards_black.append(black_episode_reward)
        all_rewards_white.append(white_episode_reward)

    return all_rewards_black, all_rewards_white, winners, turns_list


def plot_results(rew_b, rew_w, winners, turns, title="Training Results"):
    n = len(rew_b)
    window = max(1, n // 50)

    ma_rb = moving_average(rew_b, window)
    ma_rw = moving_average(rew_w, window)

    black_wins = [1 if w == 1 else 0 for w in winners]
    ma_bwins = moving_average(black_wins, window)

    ma_turns = moving_average(turns, window)

    plt.figure(figsize=(12,8))
    plt.suptitle(title)

    plt.subplot(3,1,1)
    plt.plot(ma_rb, label="Black(Reward)")
    plt.plot(ma_rw, label="White(Reward)")
    plt.legend()
    plt.ylabel("Reward(ma)")

    plt.subplot(3,1,2)
    plt.plot(ma_bwins, color="orange", label="Black win rate")
    plt.legend()
    plt.ylabel("WinRate(Black)")

    plt.subplot(3,1,3)
    plt.plot(ma_turns, color="green", label="Turns")
    plt.xlabel("Episode")
    plt.ylabel("Turns")
    plt.legend()

    plt.tight_layout()
    plt.show()


########################################################
# 4) メイン実行例
########################################################

def main():
    # ------------------------------
    # 例: ハイパーパラメータ設定
    # ------------------------------
    config = {
        "board_size": 9,
        "episodes": 2000,
        "env_params": {
            "force_center_first_move": False,
            "adjacency_range": None,  # None で制限なし
            "invalid_move_penalty": -1.0,
            "reward_chain_2_open2": 0.01,
            "reward_chain_3_open2": 0.5,
            "reward_chain_4_open2": 0.8,
            "reward_chain_2_open1": 0.0,
            "reward_chain_3_open1": 0.05,
            "reward_chain_4_open1": 0.4,
            "reward_block_2_open2": 0.05,
            "reward_block_3_open2": 0.6,
            "reward_block_4_open2": 0.0,
            "reward_block_2_open1": 0.0,
            "reward_block_3_open1": 0.05,
            "reward_block_4_open1": 0.9,
        },
        "policy_params": {
            "board_size": 9,
            "hidden_size": 128,
            "lr": 1e-3,
            "gamma": 0.95,
            "temp": 2.0,
            "min_temp": 0.5,
            "temp_decay": 0.999,
            "entropy_coef": 0.01,
        },
        "q_params": {
            "board_size": 9,
            "hidden_size": 256,
            "lr": 1e-3,
            "gamma": 0.90,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 20000,
            "replay_capacity": 100000,
            "batch_size": 64,
            "update_frequency": 10,
            "target_update_frequency": 200,
        }
    }

    # 環境作成
    env = GomokuEnv(board_size=config["board_size"], **config["env_params"])

    # ------------------------------
    # 例1: Policy vs Policy (自己対戦)
    # ------------------------------
    # black_agent = PolicyAgent(**config["policy_params"])
    # white_agent = PolicyAgent(**config["policy_params"])

    # rew_b, rew_w, winners, turns = train_agents(env, black_agent, white_agent, config["episodes"])
    # plot_results(rew_b, rew_w, winners, turns, title="Policy vs Policy")

    # black_agent.save_model(MODEL_DIR / "policy_black.pth")
    # white_agent.save_model(MODEL_DIR / "policy_white.pth")

    # ------------------------------
    # 例2: QAgent vs QAgent (自己対戦)
    # ------------------------------
    black_q = QAgent(**config["q_params"])
    white_q = QAgent(**config["q_params"])
    
    rew_b, rew_w, winners, turns = train_agents(env, black_q, white_q, config["episodes"])
    plot_results(rew_b, rew_w, winners, turns, title="Q vs Q")
    
    black_q.save_model(MODEL_DIR / "q_black.pth")
    white_q.save_model(MODEL_DIR / "q_white.pth")

    # ------------------------------
    # 例3: ヒューリスティックAgent vs 学習Agent
    #     (黒番: Policy, 白番: ImmediateWinBlock)
    # ------------------------------
    # env = GomokuEnv(board_size=config["board_size"], **config["env_params"])
    # black_policy = PolicyAgent(**config["policy_params"])
    # white_heuristic = LongestChainAgent()
    
    # rew_b, rew_w, winners, turns = train_agents(env, black_policy, white_heuristic, config["episodes"])
    # plot_results(rew_b, rew_w, winners, turns, title="Policy(Black) vs ImmediateWinBlock(White)")

    # ------------------------------
    # 例4: ヒューリスティックAgent vs 学習Agent
    #     (黒番: QAgent, 白番: FourThreePriority)
    # ------------------------------
    # env = GomokuEnv(board_size=config["board_size"], **config["env_params"])
    # black_q = QAgent(**config["q_params"])
    # white_heuristic = LongestChainAgent()
    
    # rew_b, rew_w, winners, turns = train_agents(env, black_q, white_heuristic, config["episodes"])
    # plot_results(rew_b, rew_w, winners, turns, title="Q(Black) vs FourThreePriority(White)")

    # 好みに合わせて学習させたい組み合わせを試してみてください。


def run_match_pygame(black_agent, white_agent, board_size=9, pause_time=0.5):
    """簡単なPyGame表示付き対戦実行関数"""

    # 1ゲームだけ対戦して盤面を可視化する
    env = GomokuEnv(board_size=board_size, **config["env_params"])

    # fpsは1手ごとの待ち時間の逆数
    fps = 1.0 / pause_time if pause_time > 0 else 0

    # play_with_pygame の便利関数を利用して対戦
    play_game(env, black_agent, white_agent, visualize=True, fps=fps)


if __name__ == "__main__":
    main()
    # 学習済みモデルをロードしてエージェントを再現
    black_q = QAgent(**config["q_params"])
    black_q.load_model(MODEL_DIR / "q_black.pth")
    
    white_q = QAgent(**config["q_params"])
    white_q.load_model(MODEL_DIR / "q_white.pth")

    # pygameで対戦を再生
    run_match_pygame(black_q, white_q, board_size=config["board_size"], pause_time=0.5)
