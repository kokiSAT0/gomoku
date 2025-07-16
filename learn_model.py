# learn_model.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

from gomoku_env import GomokuEnv
from utils import moving_average
from agents import (
    PolicyAgent,
    QAgent,
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent
)

# 学習済みモデルを格納するフォルダ
MODEL_DIR = Path(__file__).resolve().parent / "models"


def selfplay_policy_vs_policy(
    board_size=7,
    episodes=500,
    env_params=None,
    black_params=None,
    white_params=None
):
    """
    PolicyAgent 同士の自己対戦 (黒番:Policy, 白番:Policy)
    """
    if env_params is None:
        env_params = {}
    if black_params is None:
        black_params = {}
    if white_params is None:
        white_params = {}

    env = GomokuEnv(board_size=board_size, **env_params)
    black_agent = PolicyAgent(board_size=board_size, **black_params)
    white_agent = PolicyAgent(board_size=board_size, **white_params)

    all_rewards_black = []
    all_rewards_white = []
    winners = []
    turns_list = []

    for epi in tqdm(range(episodes), desc="Policy vs Policy"):
        obs = env.reset()
        done = False
        black_episode_reward = 0.0
        white_episode_reward = 0.0

        while not done:
            current_player = env.current_player
            if current_player == 1:
                action = black_agent.get_action(obs, env)
            else:
                action = white_agent.get_action(obs, env)

            next_obs, reward, done, info = env.step(action)

            # 報酬付与
            if current_player == 1:
                black_agent.record_reward(reward)
                black_episode_reward += reward
            else:
                white_agent.record_reward(reward)
                white_episode_reward += reward

            obs = next_obs

        black_agent.finish_episode()
        white_agent.finish_episode()

        w = info["winner"]
        winners.append(w)
        turns_list.append(env.turn_count)
        all_rewards_black.append(black_episode_reward)
        all_rewards_white.append(white_episode_reward)

    return black_agent, white_agent, all_rewards_black, all_rewards_white, winners, turns_list


def selfplay_q_vs_q(
    board_size=7,
    episodes=500,
    env_params=None,
    black_params=None,
    white_params=None
):
    """
    QAgent 同士の自己対戦 (黒番:Q, 白番:Q)
    """
    if env_params is None:
        env_params = {}
    if black_params is None:
        black_params = {}
    if white_params is None:
        white_params = {}

    env = GomokuEnv(board_size=board_size, **env_params)
    black_agent = QAgent(board_size=board_size, **black_params)
    white_agent = QAgent(board_size=board_size, **white_params)

    all_rewards_black = []
    all_rewards_white = []
    winners = []
    turns_list = []

    for epi in tqdm(range(episodes), desc="Q vs Q"):
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
            obs = next_obs

            if current_player == 1:
                black_agent.record_transition(s, action, reward, next_obs, done)
                black_episode_reward += reward
            else:
                white_agent.record_transition(s, action, reward, next_obs, done)
                white_episode_reward += reward

        black_agent.finish_episode()
        white_agent.finish_episode()

        w = info["winner"]
        winners.append(w)
        turns_list.append(env.turn_count)
        all_rewards_black.append(black_episode_reward)
        all_rewards_white.append(white_episode_reward)

    return black_agent, white_agent, all_rewards_black, all_rewards_white, winners, turns_list


def selfplay_policy_vs_q(
    board_size=7,
    episodes=500,
    env_params=None,
    black_is_policy=True,
    policy_params=None,
    q_params=None
):
    """
    PolicyAgent と QAgent を自己対戦させる。

    引数:
      - black_is_policy: True なら "黒番=Policy, 白番=Q" / False なら "黒番=Q, 白番=Policy"
      - policy_params, q_params: 各エージェントの初期化パラメータ
    戻り値: (black_agent, white_agent, all_rewards_black, all_rewards_white, winners, turns_list)
    """
    if env_params is None:
        env_params = {}
    if policy_params is None:
        policy_params = {}
    if q_params is None:
        q_params = {}

    env = GomokuEnv(board_size=board_size, **env_params)

    if black_is_policy:
        black_agent = PolicyAgent(board_size=board_size, **policy_params)
        white_agent = QAgent(board_size=board_size, **q_params)
        desc_str = "Policy(Black) vs Q(White)"
    else:
        black_agent = QAgent(board_size=board_size, **q_params)
        white_agent = PolicyAgent(board_size=board_size, **policy_params)
        desc_str = "Q(Black) vs Policy(White)"

    all_rewards_black = []
    all_rewards_white = []
    winners = []
    turns_list = []

    for epi in tqdm(range(episodes), desc=desc_str):
        obs = env.reset()
        done = False
        black_episode_reward = 0.0
        white_episode_reward = 0.0

        while not done:
            current_player = env.current_player
            s = obs.copy()

            if current_player == 1:
                # 黒番行動
                action = black_agent.get_action(obs, env)
            else:
                # 白番行動
                action = white_agent.get_action(obs, env)

            next_obs, reward, done, info = env.step(action)
            obs = next_obs

            # 報酬の記録 (PolicyAgentなら record_reward, QAgentなら record_transition)
            if current_player == 1:
                # 黒番
                if black_is_policy:
                    # Policy
                    black_agent.record_reward(reward)
                    black_episode_reward += reward
                else:
                    # Q
                    black_agent.record_transition(s, action, reward, next_obs, done)
                    black_episode_reward += reward
            else:
                # 白番
                if black_is_policy:
                    # ここは black_is_policy=True の場合 => 白番はQ
                    white_agent.record_transition(s, action, reward, next_obs, done)
                    white_episode_reward += reward
                else:
                    # ここは black_is_policy=False => 白番はPolicy
                    white_agent.record_reward(reward)
                    white_episode_reward += reward

        # エピソード終了 => finish_episode
        if black_is_policy:
            black_agent.finish_episode()
        else:
            # QAgentの場合は特に何もなし
            pass

        if black_is_policy:
            # 白番はQ => finish_episode()不要
            pass
        else:
            # 白番はPolicy => finish_episode()
            white_agent.finish_episode()

        w = info["winner"]
        winners.append(w)
        turns_list.append(env.turn_count)
        all_rewards_black.append(black_episode_reward)
        all_rewards_white.append(white_episode_reward)

    return black_agent, white_agent, all_rewards_black, all_rewards_white, winners, turns_list


def main():
    # 例1: PolicyAgent vs PolicyAgent
    black_agent, white_agent, rew_b, rew_w, winners, turns = selfplay_policy_vs_policy(
        board_size=9,
        episodes=4000,
        env_params={"force_center_first_move": False, "adjacency_range": None},
        black_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},
        white_params={"hidden_size":128, "lr":1e-3, "gamma":0.95}
    )
    plot_results(rew_b, rew_w, winners, turns, title="Policy vs Policy (9x9)")
    black_agent.save_model(MODEL_DIR / "policy_agent_black.pth")
    white_agent.save_model(MODEL_DIR / "policy_agent_white.pth")

    # 例2: QAgent vs QAgent
    # black_agent, white_agent, rew_b, rew_w, winners, turns = selfplay_q_vs_q(
    #     board_size=9,
    #     episodes=3000,
    #     env_params={"force_center_first_move": False, "adjacency_range": None},
    #     black_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},
    #     white_params={"hidden_size":128, "lr":1e-3, "gamma":0.95}
    # )
    # plot_results(rew_b, rew_w, winners, turns, title="Q vs Q (9x9)")
    # black_agent.save_model(MODEL_DIR / "q_agent_black.pth")
    # white_agent.save_model(MODEL_DIR / "q_agent_white.pth")

    # 例3: Policy(黒) vs Q(白)
    # black_agent, white_agent, rew_b, rew_w, winners, turns = selfplay_policy_vs_q(
    #     board_size=9,
    #     episodes=2000,
    #     env_params={"force_center_first_move": False, "adjacency_range": None},
    #     black_is_policy=True,
    #     policy_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},
    #     q_params={"hidden_size":128, "lr":1e-3, "gamma":0.95}
    # )
    # plot_results(rew_b, rew_w, winners, turns, title="Policy(Black) vs Q(White)")
    # black_agent.save_model(MODEL_DIR / "policy_black.pth")
    # white_agent.save_model(MODEL_DIR / "q_white.pth")

    # 例4: Q(黒) vs Policy(白)
    # black_agent, white_agent, rew_b, rew_w, winners, turns = selfplay_policy_vs_q(
    #     board_size=9,
    #     episodes=2000,
    #     env_params={"force_center_first_move": False, "adjacency_range": None},
    #     black_is_policy=False,
    #     policy_params={"hidden_size":128, "lr":1e-3, "gamma":0.95},
    #     q_params={"hidden_size":128, "lr":1e-3, "gamma":0.95}
    # )
    # plot_results(rew_b, rew_w, winners, turns, title="Q(Black) vs Policy(White)")
    # black_agent.save_model(MODEL_DIR / "q_black.pth")
    # white_agent.save_model(MODEL_DIR / "policy_white.pth")


def plot_results(rew_b, rew_w, winners, turns, title="SelfPlay"):
    import matplotlib.pyplot as plt

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
    plt.plot(ma_bwins, color="orange")
    plt.ylabel("WinRate(Black)")

    plt.subplot(3,1,3)
    plt.plot(ma_turns, color="green")
    plt.xlabel("Episode")
    plt.ylabel("Turns")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
