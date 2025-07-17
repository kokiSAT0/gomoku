# play_with_pygame.py

"""Pygame を用いて GUI 上で対戦を可視化するスクリプト。

学習済みモデルやヒューリスティックエージェントを組み合わせ、
1 ゲームのみ盤面を表示しながらプレイする際に使用する。
"""

import pygame
import sys
import time
from pathlib import Path

# 新しいパッケージ構成に合わせてインポート
from ..core.gomoku_env import GomokuEnv
from .pygame_utils import draw_board, CELL_SIZE
from .agent_factory import create_agent
from .result_utils import show_results

# 学習済みモデルをまとめたフォルダ
# 上位ディレクトリに配置されている ``models`` フォルダを取得
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

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
    # --- 複数対局を実行 -------------------------------------------------
    winners = []
    black_rewards = []
    turn_counts = []

    for game_idx in range(num_games):
        env = GomokuEnv(board_size=board_size, **env_params)
        # 最初のゲームのみ表示する
        do_visual = visualize and (game_idx == 0)

        winner, black_reward, turn_count = play_game(
            env=env,
            black_agent=black_agent,
            white_agent=white_agent,
            visualize=do_visual,
            fps=fps,
        )
        winners.append(winner)
        black_rewards.append(black_reward)
        turn_counts.append(turn_count)

    # --- 結果表示 -------------------------------------------------------
    show_results(
        black_agent_type=black_agent_type,
        black_agent_path=black_agent_path,
        white_agent_type=white_agent_type,
        white_agent_path=white_agent_path,
        winners=winners,
        black_rewards=black_rewards,
        turn_counts=turn_counts,
    )


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
        black_agent_path=MODEL_DIR / "policy_agent_black.pth",  # 学習済みモデルファイル
        black_agent_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},

        white_agent_type="policy",
        white_agent_path=MODEL_DIR / "policy_agent_white.pth",  # ヒューリスティックならNone
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
    #     black_agent_path=MODEL_DIR / "q_agent_black.pth",
    #     black_agent_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},
    #
    #     white_agent_type="longest",
    #     white_agent_path=None,
    #     white_agent_params={},
    #
    #     visualize=True,  # 最初の試合のみ表示
    #     fps=2
    # )
