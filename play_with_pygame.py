# play_with_pygame.py

import pygame
import sys
import time
import numpy as np

from gomoku_env import GomokuEnv
from agents import (
    PolicyAgent,
    QAgent,
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent
)

CELL_SIZE = 50                # 1マスあたりのピクセルサイズ
LINE_COLOR = (0, 0, 0)        # グリッド線(黒)
BG_COLOR = (222, 184, 135)    # 碁盤っぽい背景色 (ベージュ)
BLACK_STONE_COLOR = (0, 0, 0) # 黒石
WHITE_STONE_COLOR = (255, 255, 255) # 白石


def draw_grid(screen, board_size):
    """碁盤のグリッド線を描画するヘルパー"""
    for i in range(board_size):
        # 縦線の描画
        start_pos = (i * CELL_SIZE + CELL_SIZE // 2, CELL_SIZE // 2)
        end_pos = (
            i * CELL_SIZE + CELL_SIZE // 2,
            (board_size - 1) * CELL_SIZE + CELL_SIZE // 2,
        )
        pygame.draw.line(screen, LINE_COLOR, start_pos, end_pos, 2)

        # 横線の描画
        start_pos = (CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2)
        end_pos = (
            (board_size - 1) * CELL_SIZE + CELL_SIZE // 2,
            i * CELL_SIZE + CELL_SIZE // 2,
        )
        pygame.draw.line(screen, LINE_COLOR, start_pos, end_pos, 2)


def draw_stones(screen, board):
    """盤面上の石を全て描画する"""
    board_size = board.shape[0]

    # 石のある場所だけを抽出してループする
    for x, y in np.argwhere(board != 0):
        stone = board[x, y]
        center_pos = (
            y * CELL_SIZE + CELL_SIZE // 2,
            x * CELL_SIZE + CELL_SIZE // 2,
        )
        color = BLACK_STONE_COLOR if stone == 1 else WHITE_STONE_COLOR
        pygame.draw.circle(screen, color, center_pos, CELL_SIZE // 2 - 2)


def draw_board(screen, env):
    """背景・グリッド・石をまとめて描画するメイン関数"""
    board_size = env.board_size
    screen.fill(BG_COLOR)
    draw_grid(screen, board_size)
    draw_stones(screen, env.game.board)


def play_game(env, black_agent, white_agent, visualize=True, fps=1):
    """
    1ゲームだけ実行して (winner, black_reward, turn_count) を返す。
      - visualize=True なら PyGame で碁盤を描画しながら進行
      - 黒番視点の報酬(勝ち=+1,負け=-1,引き分け=0)を返す
    """
    obs = env.reset()
    done = False

    # PyGameウィンドウの初期化
    if visualize:
        pygame.init()
        board_px_size = env.board_size * CELL_SIZE
        screen = pygame.display.set_mode((board_px_size, board_px_size))
        pygame.display.set_caption("Gomoku Visualization")
        clock = pygame.time.Clock()

    while not done:
        if visualize:
            # ウィンドウのイベント処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # 画面描画
            draw_board(screen, env)
            pygame.display.flip()
            clock.tick(fps)

        # 手番のプレイヤーが行動を選択
        if env.current_player == 1:
            action = black_agent.get_action(obs, env)
        else:
            action = white_agent.get_action(obs, env)

        obs, reward, done, info = env.step(action)

    # 最終盤面を描画して少し待機
    if visualize:
        draw_board(screen, env)
        pygame.display.flip()
        time.sleep(1)
        pygame.quit()

    # 決着情報を集計
    winner = info["winner"]  # 1=黒, 2=白, -1=引き分け
    if winner == 1:
        black_reward = 1.0
    elif winner == 2:
        black_reward = -1.0
    else:
        black_reward = 0.0
    turn_count = env.turn_count

    return winner, black_reward, turn_count


# ------------------------------------------------------------------------------
# メインの対戦関数: 黒番, 白番それぞれを自由に設定し、複数回実行して結果を出力
# ------------------------------------------------------------------------------
def play_agents_vs_agents(
    board_size=9,
    num_games=1,
    env_params=None,
    # 黒番の指定
    black_agent_type="policy",    # "policy", "q", "random", "immediate", "fourthree", "longest" etc.
    black_agent_path=None,        # 学習済みモデルファイル(pth)があれば
    black_agent_params=None,      # 初期化時パラメータ
    # 白番の指定
    white_agent_type="LongestChainAgent",
    white_agent_path=None,
    white_agent_params=None,
    # 可視化
    visualize=True,
    fps=1
):
    """
    黒番・白番のエージェントを好きな組合せで対戦させる。
      - black_agent_type, white_agent_type: エージェント種類("policy"/"q"/"random"など)
      - black_agent_path, white_agent_path: 学習済みモデルファイルパス (policy/qの場合のみロード)
      - black_agent_params, white_agent_params: その他パラメータ(dict)
      - num_games: 試合数 (1が推奨。>1で可視化するとウィンドウが開き直す)
      - visualize: TrueならPyGameで表示 (基本的に最初の1試合のみ表示)
      - fps: 表示速度
    """
    if env_params is None:
        env_params = {}
    if black_agent_params is None:
        black_agent_params = {}
    if white_agent_params is None:
        white_agent_params = {}

    # --- 黒番エージェントを作成 ---
    black_agent = create_agent(
        agent_type=black_agent_type,
        board_size=board_size,
        agent_path=black_agent_path,
        agent_params=black_agent_params
    )

    # --- 白番エージェントを作成 ---
    white_agent = create_agent(
        agent_type=white_agent_type,
        board_size=board_size,
        agent_path=white_agent_path,
        agent_params=white_agent_params
    )

    # 複数対局を実行
    winners = []
    black_rewards = []
    turn_counts = []

    for game_idx in range(num_games):
        env = GomokuEnv(board_size=board_size, **env_params)
        # 最初のゲームだけ表示する
        do_visual = visualize and (game_idx == 0)

        winner, black_reward, turn_count = play_game(
            env=env,
            black_agent=black_agent,
            white_agent=white_agent,
            visualize=do_visual,
            fps=fps
        )
        winners.append(winner)
        black_rewards.append(black_reward)
        turn_counts.append(turn_count)

    # 集計結果を表示
    show_results(
        black_agent_type=black_agent_type,
        black_agent_path=black_agent_path,
        white_agent_type=white_agent_type,
        white_agent_path=white_agent_path,
        winners=winners,
        black_rewards=black_rewards,
        turn_counts=turn_counts
    )

# ------------------------------------------------------------------------------
# エージェント生成用ヘルパー
# ------------------------------------------------------------------------------
def create_agent(agent_type, board_size, agent_path=None, agent_params=None):
    """
    agent_type の文字列に応じてエージェントを生成するヘルパー関数。

    - "policy"/"q" などモデルを要するものはファイル読み込みにも対応。
    - 該当しない文字列が渡された場合は RandomAgent を返す。
    """

    if agent_params is None:
        agent_params = {}

    # 文字列を小文字化しておく
    key = agent_type.lower()

    # 同義語をまとめる辞書
    alias = {
        "rand": "random",
        "immediatewinblockagent": "immediate",
        "fourthreepriorityagent": "fourthree",
        "longestchainagent": "longest",
    }
    key = alias.get(key, key)

    # エージェントクラスへのマッピング
    class_table = {
        "policy": PolicyAgent,
        "q": QAgent,
        "random": RandomAgent,
        "immediate": ImmediateWinBlockAgent,
        "fourthree": FourThreePriorityAgent,
        "longest": LongestChainAgent,
    }

    if key in ("policy", "q"):
        # 学習済みモデルを読み込む可能性があるエージェント
        AgentClass = class_table[key]
        agent = AgentClass(board_size=board_size, **agent_params)
        if agent_path:
            agent.load_model(agent_path)
        return agent

    if key in class_table:
        # ヒューリスティック系 / RandomAgent
        AgentClass = class_table[key]
        return AgentClass()

    # 万が一該当しなければ RandomAgent で代替
    print(f"[WARNING] Unknown agent_type={agent_type}, fallback to RandomAgent.")
    return RandomAgent()


# ------------------------------------------------------------------------------
# 結果表示用ヘルパー
# ------------------------------------------------------------------------------
def show_results(
    black_agent_type,
    black_agent_path,
    white_agent_type,
    white_agent_path,
    winners,
    black_rewards,
    turn_counts
):
    """
    複数試合の結果を集計してprint。
    """
    import numpy as np

    num_games = len(winners)
    avg_reward = float(np.mean(black_rewards))
    black_win_rate = float(np.mean([1.0 if w == 1 else 0.0 for w in winners]))
    avg_turn_count = float(np.mean(turn_counts))

    print("============================================================")
    print(f"対局数: {num_games}")
    print(f"黒番 = {black_agent_type}, path={black_agent_path}")
    print(f"白番 = {white_agent_type}, path={white_agent_path}")
    print(f"黒番 平均報酬(視点) : {avg_reward:.3f}")
    print(f"黒番 勝率          : {black_win_rate:.3f}")
    print(f"平均決着手数       : {avg_turn_count:.1f}")
    print("------------------------------------------------------------")

    show_count = min(num_games, 10)
    print(f"(先頭{show_count}ゲームの結果) => (winner, black_reward, turn)")
    for i in range(show_count):
        print(f"Game {i+1:2d}: Winner={winners[i]}, RewardForBlack={black_rewards[i]}, Turn={turn_counts[i]}")
    print("============================================================")


# ------------------------------------------------------------------------------
# 実行例
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    サンプル1:
      - 黒番: 学習済みPolicyAgent (policy_agent_black.pthをロード)
      - 白番: ヒューリスティックImmediateWinBlockAgent
      - 1ゲームだけ可視化
    """
    play_agents_vs_agents(
        board_size=9,
        num_games=1,
        env_params={"force_center_first_move": False, "adjacency_range": 1},

        black_agent_type="policy",
        black_agent_path="policy_agent_black.pth",  # 学習済みモデルファイル
        black_agent_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},

        white_agent_type="policy",
        white_agent_path="policy_agent_white.pth",  # ヒューリスティックならNone
        white_agent_params={},

        visualize=True,
        fps=2
    )

    """
    サンプル2:
      - 黒番: QAgent (学習済み q_agent_black.pth)
      - 白番: LongestChainAgent (ヒューリスティック)
      - 5ゲーム連続で対戦 (1ゲーム目のみ可視化)
    """
    # play_agents_vs_agents(
    #     board_size=9,
    #     num_games=5,
    #     env_params={"force_center_first_move": True, "adjacency_range": 1},
    #
    #     black_agent_type="q",
    #     black_agent_path="q_agent_black.pth",
    #     black_agent_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},
    #
    #     white_agent_type="longest",
    #     white_agent_path=None,
    #     white_agent_params={},
    #
    #     visualize=True,  # 最初の試合のみ表示
    #     fps=2
    # )
