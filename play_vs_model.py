# play_vs_model.py
import torch
import numpy as np

from gomoku_env import GomokuEnv
from agents import PolicyAgent, RandomAgent

def play_game_with_trained_model(policy_path="policy_agent_trained.pth"):
    # 学習済みモデルを読み込んだPolicyAgent(黒番)とRandomAgent(白番)で対戦
    env = GomokuEnv(board_size=9)
    black_agent = PolicyAgent(board_size=9)
    black_agent.load_model(policy_path)
    
    white_agent = RandomAgent()
    
    obs = env.reset()
    done = False
    
    while not done:
        if env.current_player == 1:  # 黒
            action = black_agent.get_action(obs, env)
        else:  # 白
            action = white_agent.get_action(obs, env)
        
        obs, reward, done, info = env.step(action)
        env.render()
    
    winner = info["winner"]
    if winner == 1:
        print("黒番(PolicyAgent)の勝ち！")
    elif winner == 2:
        print("白番(RandomAgent)の勝ち！")
    else:
        print("引き分け")

if __name__ == "__main__":
    play_game_with_trained_model("policy_agent_trained.pth")
