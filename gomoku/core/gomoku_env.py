# 同一パッケージ内のユーティリティを相対インポート
from .game import Gomoku, count_chains_open_ends
from .utils import opponent_player
from .reward_utils import compute_rewards
# can_place_stone など細かな判定処理は env_utils へ切り出した
from .env_utils import can_place_stone


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
        # 2手目以降、既存の石から adjacency_range 以内かどうか (コメントアウトで無効化)
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
        """action 番号を ``(x, y)`` 座標へ変換するヘルパー"""
        x = action // self.board_size
        y = action % self.board_size
        return x, y

    def coord_to_action(self, x: int, y: int) -> int:
        """``(x, y)`` 座標を action 番号へ変換するヘルパー"""
        return x * self.board_size + y

    def can_place_stone(self, x, y):
        """env_utils に定義した判定関数へのラッパー"""
        return can_place_stone(self, x, y)


    def step(self, action):
        """
        ``action`` は ``0`` から ``board_size**2 - 1`` の整数。
        ``action_to_coord()`` を用いて盤面座標 ``(x, y)`` に変換して着手する。
        戻り値: ``(obs, reward, done, info)``
        """
        if self.done:
            # 既に終わっている場合、報酬0で終了状態を返す
            return self._get_observation(), 0.0, True, {"winner": self.winner}

        # 現在のプレイヤーと相手プレイヤーを取得
        current_player = self.game.current_player
        # 相手プレイヤーIDを取得
        opponent = opponent_player(current_player)

        # 着手前の連数を計測
        before_self = count_chains_open_ends(self.game.board, current_player)
        before_opp = count_chains_open_ends(self.game.board, opponent)

        # 座標に変換
        x, y = self.action_to_coord(action)

        # 1. 無効手チェック
        if not self.can_place_stone(x, y):
            self.done = True
            self.winner = 0  # 勝者なし
            return self._get_observation(), self.invalid_move_penalty, True, {
                "winner": 0, "reason": "invalid_move"
            }

        # 2. 石を置く処理
        self.game.place_stone(x, y)
        self.turn_count += 1

        # 3. 勝敗判定と中間報酬の計算
        total_reward, self.done, self.winner = compute_rewards(
            game=self.game,
            current_player=current_player,
            opponent=opponent,
            before_self=before_self,
            before_opp=before_opp,
            r_chain=self.r_chain,
            r_block=self.r_block,
        )

        return self._get_observation(), total_reward, self.done, {
            "winner": self.winner
        }

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
