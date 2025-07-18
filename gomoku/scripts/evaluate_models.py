"""学習済みモデルの性能を評価するスクリプト。

指定したエージェント(PolicyAgent または QAgent)をヒューリスティック
エージェントと複数回対戦させ、黒番の勝率を計算する。
例:
    $ python evaluate_models.py
"""

from pathlib import Path
import argparse
from ..core.gomoku_env import GomokuEnv
from ..ai.agents import (
    PolicyAgent,
    QAgent,
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent,
)  # 自作のエージェントクラスをインポート

# 学習済みモデルを配置しているフォルダへのパス
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

def evaluate_model(
    policy_path=MODEL_DIR / "policy_agent_trained.pth",
    q_path=MODEL_DIR / "q_agent.pth",
    opponent_agent=None,
    num_episodes=1000,
    board_size=9,
    device=None,
    policy_color="black",
    network_type="dense",
    q_network_type="fc",
    eval_temp=0.5,
    agent_type="policy",
):
    """
    保存済みモデルを読み込み、指定した相手と複数回対戦させて
    勝率を返す。``agent_type`` で ``PolicyAgent`` か ``QAgent`` を選択できる。

    引数:
      policy_path: PolicyAgent 用モデルのパス
      q_path: QAgent 用モデルのパス
      opponent_agent: 対戦相手(Agent のインスタンス)
      num_episodes: 対戦回数
      board_size: 盤面サイズ
      device: 使用デバイス ("cuda" / "cpu" など)
      policy_color: "black" なら先手、"white" なら後手として評価
      eval_temp: 評価時に用いる temperature 値
      agent_type: 評価するエージェントの種類 ("policy" or "q")

    戻り値:
      win_rate: 指定エージェントの勝率 (0.0 ~ 1.0)
    """
    # 1) 評価対象のエージェントを読み込み
    if agent_type == "policy":
        eval_agent = PolicyAgent(
            board_size=board_size, device=device, network_type=network_type
        )
        eval_agent.load_model(policy_path)
        eval_agent.model.eval()
        eval_agent.temp = eval_temp
    else:
        eval_agent = QAgent(
            board_size=board_size, device=device, network_type=q_network_type
        )
        eval_agent.load_model(q_path)

    def play_single_game(black, white) -> int:
        """1 試合だけ実行し勝者を返す"""
        env = GomokuEnv(board_size=board_size)
        obs = env.reset()
        done = False

        while not done:
            if env.current_player == 1:
                action = black.get_action(obs, env)
            else:
                action = white.get_action(obs, env)
            obs, _, done, info = env.step(action)

        return info["winner"]

    # 2) 対戦相手を用意 (引数で与えられなければ RandomAgent とする)
    if opponent_agent is None:
        opponent_agent = RandomAgent()

    # 3) 複数回対戦して勝率を測る
    wins = 0
    draws = 0
    losses = 0

    for _ in range(num_episodes):
        if policy_color == "black":
            winner = play_single_game(eval_agent, opponent_agent)
        else:
            winner = play_single_game(opponent_agent, eval_agent)

        # 対局終了後に勝敗を確認 (eval_agent 視点)
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
    label = "PolicyAgent" if agent_type == "policy" else "QAgent"
    print(f"{color_label}({label}) 勝ち: {wins}, 負け: {losses}, 引き分け: {draws}")
    print(f"勝率: {win_rate:.2f}")

    return win_rate

if __name__ == "__main__":
    # コマンドライン引数を利用して設定を受け取る
    parser = argparse.ArgumentParser(
        description="学習済みモデルの評価を行うユーティリティ"
    )
    parser.add_argument(
        "--policy_path",
        type=Path,
        default=MODEL_DIR / "policy_agent_trained.pth",
        help="PolicyAgent 用モデルファイルのパス",
    )
    parser.add_argument(
        "--q_path",
        type=Path,
        default=MODEL_DIR / "q_agent.pth",
        help="QAgent 用モデルファイルのパス",
    )
    parser.add_argument(
        "--agent_type",
        choices=["policy", "q"],
        default="policy",
        help="評価対象エージェントの種類",
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
        help="評価対象エージェントを先手または後手どちらで用いるか",
    )
    parser.add_argument(
        "--eval_temp", type=float, default=0.5, help="評価時の temperature"
    )
    parser.add_argument(
        "--network_type",
        choices=["dense", "conv"],
        default="dense",
        help="PolicyAgent のネットワーク形式",
    )
    parser.add_argument(
        "--q_network_type",
        choices=["fc", "conv"],
        default="fc",
        help="QAgent のネットワーク形式",
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
        q_path=args.q_path,
        opponent_agent=opp,
        num_episodes=args.num_episodes,
        board_size=args.board_size,
        device=args.device,
        policy_color=args.policy_color,
        network_type=args.network_type,
        q_network_type=args.q_network_type,
        eval_temp=args.eval_temp,
        agent_type=args.agent_type,
    )
