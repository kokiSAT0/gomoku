"""ゲーム実行用のユーティリティ関数群"""

from pathlib import Path
import sys
import time
import os
import warnings

from ..core.gomoku_env import GomokuEnv
from .pygame_utils import draw_board, CELL_SIZE
from .agent_factory import create_agent
from .result_utils import show_results

# 学習済みモデルの保存ディレクトリ
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def play_game(env: GomokuEnv, black_agent, white_agent, visualize: bool = True, fps: int = 1):
    """1 ゲームだけ実行して結果を返す関数"""
    obs = env.reset()
    done = False

    # Pygame を利用する場合のみインポートする
    if visualize:
        # "Hello from the pygame community" メッセージを抑制する
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")

        # pkg_resources に関する警告も非表示にする
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API",
            category=UserWarning,
        )

        import pygame

        pygame.init()
        board_px_size = env.board_size * CELL_SIZE
        screen = pygame.display.set_mode((board_px_size, board_px_size))
        pygame.display.set_caption("Gomoku Visualization")
        clock = pygame.time.Clock()
    
    while not done:
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            draw_board(screen, env)
            pygame.display.flip()
            clock.tick(fps)

        if env.current_player == 1:
            action = black_agent.get_action(obs, env)
        else:
            action = white_agent.get_action(obs, env)

        obs, reward, done, info = env.step(action)

    if visualize:
        draw_board(screen, env)
        pygame.display.flip()
        time.sleep(1)
        pygame.quit()

    winner = info["winner"]
    if winner == 1:
        black_reward = 1.0
    elif winner == 2:
        black_reward = -1.0
    else:
        black_reward = 0.0
    turn_count = env.turn_count

    return winner, black_reward, turn_count


def play_game_text(env: GomokuEnv, black_agent, white_agent, pause: float = 0.0):
    """テキストのみで 1 ゲームを実行し盤面を逐次表示する"""

    # --- 環境を初期化し最初の盤面を表示 -----------------------------
    obs = env.reset()
    done = False
    env.render()

    # --- ゲームが終わるまで行動と描画を繰り返す ---------------------
    while not done:
        # 手番に応じてエージェントを切り替える
        if env.current_player == 1:
            action = black_agent.get_action(obs, env)
        else:
            action = white_agent.get_action(obs, env)

        # 行動を適用し盤面を更新
        obs, reward, done, info = env.step(action)

        # 現在の盤面を表示
        env.render()
        if pause > 0:
            time.sleep(pause)

    # --- 勝者を判定し情報を返す -----------------------------------
    winner = info["winner"]
    if winner == 1:
        black_reward = 1.0
    elif winner == 2:
        black_reward = -1.0
    else:
        black_reward = 0.0
    turn_count = env.turn_count

    return winner, black_reward, turn_count


def play_agents_vs_agents(
    board_size: int = 9,
    num_games: int = 1,
    env_params: dict | None = None,
    black_agent_type: str = "policy",
    black_agent_path=None,
    black_agent_params: dict | None = None,
    white_agent_type: str = "LongestChainAgent",
    white_agent_path=None,
    white_agent_params: dict | None = None,
    visualize: bool = True,
    fps: int = 1,
):
    """任意のエージェント同士を対戦させるメイン関数"""
    if env_params is None:
        env_params = {}
    if black_agent_params is None:
        black_agent_params = {}
    if white_agent_params is None:
        white_agent_params = {}

    black_agent = create_agent(
        agent_type=black_agent_type,
        board_size=board_size,
        agent_path=black_agent_path,
        agent_params=black_agent_params,
    )

    white_agent = create_agent(
        agent_type=white_agent_type,
        board_size=board_size,
        agent_path=white_agent_path,
        agent_params=white_agent_params,
    )

    winners: list[int] = []
    black_rewards: list[float] = []
    turn_counts: list[int] = []

    for game_idx in range(num_games):
        env = GomokuEnv(board_size=board_size, **env_params)
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

    show_results(
        black_agent_type=black_agent_type,
        black_agent_path=black_agent_path,
        white_agent_type=white_agent_type,
        white_agent_path=white_agent_path,
        winners=winners,
        black_rewards=black_rewards,
        turn_counts=turn_counts,
    )


__all__ = ["play_game", "play_game_text", "play_agents_vs_agents"]

