"""学習済みモデルの性能を評価するスクリプト。

指定した ``PolicyAgent`` を他のエージェントと複数回対戦させ、
黒番の勝率を計算する。
例:
    $ python evaluate_models.py
"""

from pathlib import Path
import argparse
from ..core.gomoku_env import GomokuEnv
from ..ai.agents import (
    PolicyAgent,
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent,
)  # 自作のエージェントクラスをインポート

# 学習済みモデルを配置しているフォルダへのパス
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

def evaluate_model(
    policy_path=MODEL_DIR / "policy_agent_trained.pth",
    opponent_agent=None,
    num_episodes=1000,
    board_size=9,
    device=None,
    policy_color="black",
    eval_temp=0.5,
):
    """
    保存済みモデル(PolicyAgent)を読み込み、指定した相手と複数回対戦させて
    PolicyAgent の勝率を返す。``policy_color`` で先手/後手を指定できる。

    引数:
      policy_path: 保存済みモデルのパス
      opponent_agent: 対戦相手(Agentのインスタンス)
      num_episodes: 対戦回数
      board_size: 盤面サイズ
      device: 使用デバイス ("cuda" / "cpu" など)
      policy_color: "black" なら先手、"white" なら後手として評価
      eval_temp: 評価時に用いる temperature 値

    戻り値:
      win_rate: PolicyAgent の勝率 (0.0 ~ 1.0)
    """
    # 1) 評価対象のPolicyAgentを読み込み
    policy_agent = PolicyAgent(board_size=board_size, device=device)
    policy_agent.load_model(policy_path)  # 事前に学習済みモデルを読み込む
    policy_agent.model.eval()  # 評価モード
    # 評価時の探索ノイズの大きさを指定
    policy_agent.temp = eval_temp

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
            if env.current_player == 1:
                # 先手の手番
                if policy_color == "black":
                    action = policy_agent.get_action(obs, env)
                else:
                    action = opponent_agent.get_action(obs, env)
            else:
                # 後手の手番
                if policy_color == "white":
                    action = policy_agent.get_action(obs, env)
                else:
                    action = opponent_agent.get_action(obs, env)

            obs, reward, done, info = env.step(action)

        # 対局終了後に勝敗を確認 (policy_agent 視点)
        winner = info["winner"]
        if policy_color == "black":
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1
        else:  # policy_color == "white"
            if winner == 2:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1

    # 4) 成績表示
    win_rate = wins / num_episodes
    print(f"対戦回数: {num_episodes}")
    color_label = "先手" if policy_color == "black" else "後手"
    print(f"{color_label}(PolicyAgent) 勝ち: {wins}, 負け: {losses}, 引き分け: {draws}")
    print(f"勝率: {win_rate:.2f}")

    return win_rate

if __name__ == "__main__":
    # コマンドライン引数を利用して設定を受け取る
    parser = argparse.ArgumentParser(
        description="学習済み PolicyAgent の評価を行うユーティリティ"
    )
    parser.add_argument(
        "--policy_path",
        type=Path,
        default=MODEL_DIR / "policy_agent_trained.pth",
        help="評価に用いる PolicyAgent のモデルファイルパス",
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "immediate", "fourthree", "longest"],
        default="random",
        help="対戦相手となるエージェントの種類",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="対戦回数"
    )
    parser.add_argument(
        "--board_size", type=int, default=9, help="盤面のサイズ"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="使用するデバイス名"
    )
    parser.add_argument(
        "--policy_color",
        choices=["black", "white"],
        default="black",
        help="PolicyAgent を先手または後手で評価するかを指定",
    )
    parser.add_argument(
        "--eval_temp", type=float, default=0.5, help="評価時の temperature"
    )

    args = parser.parse_args()

    # 指定された文字列から相手エージェントを生成
    if args.opponent == "random":
        opp = RandomAgent()
    elif args.opponent == "immediate":
        opp = ImmediateWinBlockAgent()
    elif args.opponent == "fourthree":
        opp = FourThreePriorityAgent()
    else:  # "longest"
        opp = LongestChainAgent()

    evaluate_model(
        policy_path=args.policy_path,
        opponent_agent=opp,
        num_episodes=args.num_episodes,
        board_size=args.board_size,
        device=args.device,
        policy_color=args.policy_color,
        eval_temp=args.eval_temp,
    )
