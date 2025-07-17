# -*- coding: utf-8 -*-
"""学習用のユーティリティ関数をまとめたモジュール"""

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..core.gomoku_env import GomokuEnv
from ..core.utils import moving_average, FIGURE_DIR
# play_with_pygame.py にはサンプル用の実行コードのみ存在するため
# 実際のゲームロジックを提供する play_utils.py から ``play_game`` を読み込む
from .play_utils import play_game, play_game_text


def train_agents(
    env,
    black_agent,
    white_agent,
    episodes: int = 1000,
    eval_interval: int = 0,
    eval_pause: float = 0.0,
):
    """自己対戦によりエージェントを学習させる汎用ループ

    Parameters
    ----------
    env : GomokuEnv
        学習に使用する環境インスタンス
    black_agent : Any
        黒番のエージェント
    white_agent : Any
        白番のエージェント
    episodes : int, optional
        学習エピソード数
    eval_interval : int, optional
        ``eval_interval`` > 0 のとき、この間隔でテキスト対戦を表示する
    eval_pause : float, optional
        テキスト対戦表示時の待ち時間 (秒)
    """
    all_rewards_black = []
    all_rewards_white = []
    winners = []
    turns_list = []

    for ep in tqdm(range(episodes), desc="train_agents"):
        obs = env.reset()
        done = False
        black_episode_reward = 0.0
        white_episode_reward = 0.0

        while not done:
            current_player = env.current_player
            state = obs.copy()

            if current_player == 1:
                action = black_agent.get_action(obs, env)
            else:
                action = white_agent.get_action(obs, env)

            next_obs, reward, done, info = env.step(action)

            if current_player == 1:
                black_agent.record_reward(reward)
                black_agent.record_transition(state, action, reward, next_obs, done)
                black_episode_reward += reward
            else:
                white_agent.record_reward(reward)
                white_agent.record_transition(state, action, reward, next_obs, done)
                white_episode_reward += reward

            obs = next_obs

        winner = info["winner"]
        if winner == 1:
            white_agent.record_reward(-1.0)
        elif winner == 2:
            black_agent.record_reward(-1.0)
        elif winner == -1:
            white_agent.record_reward(0.0)
            black_agent.record_reward(0.0)

        black_agent.finish_episode()
        white_agent.finish_episode()

        winners.append(winner)
        turns_list.append(env.turn_count)
        all_rewards_black.append(black_episode_reward)
        all_rewards_white.append(white_episode_reward)

        # --- 指定エピソード毎にテキスト対戦を実行 ------------------
        if eval_interval > 0 and (ep + 1) % eval_interval == 0:
            print(f"\n[評価] エピソード {ep + 1} 時点での対戦例")
            play_game_text(env, black_agent, white_agent, pause=eval_pause)

    return all_rewards_black, all_rewards_white, winners, turns_list


def plot_results(rew_b, rew_w, winners, turns, title="Training Results", show=True):
    """学習曲線をグラフ化して ``FIGURE_DIR`` へ保存する"""
    n = len(rew_b)
    window = max(1, n // 50)

    ma_rb = moving_average(rew_b, window)
    ma_rw = moving_average(rew_w, window)

    black_wins = [1 if w == 1 else 0 for w in winners]
    ma_bwins = moving_average(black_wins, window)

    ma_turns = moving_average(turns, window)

    plt.figure(figsize=(12, 8))
    plt.suptitle(title)

    plt.subplot(3, 1, 1)
    plt.plot(ma_rb, label="Black(Reward)")
    plt.plot(ma_rw, label="White(Reward)")
    plt.legend()
    plt.ylabel("Reward(ma)")

    plt.subplot(3, 2, 3)
    plt.plot(ma_bwins, color="orange", label="Black win rate")
    plt.legend()
    plt.ylabel("WinRate(Black)")

    plt.subplot(3, 1, 3)
    plt.plot(ma_turns, color="green", label="Turns")
    plt.xlabel("Episode")
    plt.ylabel("Turns")
    plt.legend()

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title.replace(' ', '_')}_{timestamp}.png"
    save_path = Path(FIGURE_DIR) / filename
    plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def run_match_pygame(black_agent, white_agent, board_size=9, pause_time=0.5, env_params=None):
    """PyGame を用いて 1 ゲームだけ対戦を可視化する"""
    if env_params is None:
        env_params = {}

    env = GomokuEnv(board_size=board_size, **env_params)
    fps = 1.0 / pause_time if pause_time > 0 else 0
    play_game(env, black_agent, white_agent, visualize=True, fps=fps)
