"""学習済みモデルの性能を評価するスクリプト。

指定した ``PolicyAgent`` を他のエージェントと複数回対戦させ、
黒番の勝率を計算する。
例:
    $ python evaluate_models.py
"""

from pathlib import Path
from gomoku_env import GomokuEnv
from agents import (
    PolicyAgent,
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent,
)  # 自作のエージェントクラスをインポート

# 学習済みモデルを配置しているフォルダへのパス
MODEL_DIR = Path(__file__).resolve().parent / "models"

def evaluate_model(
    policy_path=MODEL_DIR / "policy_agent_trained.pth",
    opponent_agent=None,
    num_episodes=1000,
    board_size=9
):
    """
    保存済みのモデル(PolicyAgent)を読み込み、opponent_agent と複数回対戦させて
    黒番(PolicyAgent)の勝率を返す。

    引数:
      policy_path: 保存済みモデルのパス
      opponent_agent: 対戦相手(Agentのインスタンス)
      num_episodes: 対戦回数
      board_size: 盤面サイズ

    戻り値:
      win_rate: 黒番の勝率 (0.0 ~ 1.0)
    """
    # 1) 評価対象のPolicyAgentを読み込み
    black_agent = PolicyAgent(board_size=board_size)
    black_agent.load_model(policy_path)  # 事前に学習済みモデルを読み込む
    black_agent.model.eval()  # 評価モード

    # 2) 対戦相手を用意 (引数で与えられなければ RandomAgent とする)
    if opponent_agent is None:
        opponent_agent = RandomAgent()

    # 3) 複数回対戦して勝率を測る
    wins = 0
    draws = 0
    losses = 0

    for _ in range(num_episodes):
        env = GomokuEnv(board_size=board_size)
        obs = env.reset()
        done = False

        while not done:
            if env.current_player == 1:  # 黒番(PolicyAgent)
                action = black_agent.get_action(obs, env)
            else:  # 白番(対戦相手)
                action = opponent_agent.get_action(obs, env)

            obs, reward, done, info = env.step(action)

        # 対局終了後に勝敗を確認
        winner = info["winner"]
        if winner == 1:   # 黒番勝利
            wins += 1
        elif winner == 2: # 白番勝利
            losses += 1
        else:             # 引き分け
            draws += 1

    # 4) 成績表示
    win_rate = wins / num_episodes
    print(f"対戦回数: {num_episodes}")
    print(f"黒番(PolicyAgent) 勝ち: {wins}, 負け: {losses}, 引き分け: {draws}")
    print(f"勝率: {win_rate:.2f}")

    return win_rate

if __name__ == "__main__":
    # 例: ランダムエージェントと100回対戦
    evaluate_model(
        policy_path=MODEL_DIR / "policy_agent_trained.pth",
        opponent_agent=FourThreePriorityAgent(),
        num_episodes=100,
        board_size=9
    )
