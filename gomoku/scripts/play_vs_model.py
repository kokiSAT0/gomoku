# play_vs_model.py

"""端末上で学習済みモデルと対戦するための簡易インタフェース。

黒番に ``PolicyAgent`` を読み込み、白番は ``RandomAgent`` との対戦を
1 ゲームだけ実行する。コマンドラインから次のように利用する。
    $ python play_vs_model.py
"""
from pathlib import Path

# パッケージ構成変更に伴いインポートパスを修正
from ..core.gomoku_env import GomokuEnv
from ..ai.agents import PolicyAgent, RandomAgent

# 学習済みモデルを保存したフォルダ
# スクリプトディレクトリから二階層上の ``models`` フォルダを参照
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

def play_game_with_trained_model(
    policy_path=MODEL_DIR / "policy_agent_trained.pth", network_type="dense"
):
    """
    学習済みモデルを用いて1ゲームだけ対戦を行う関数。

    Parameters
    ----------
    policy_path : str
        黒番の ``PolicyAgent`` が読み込むモデルファイルのパス。

    Returns
    -------
    None
        対局結果を標準出力に表示するのみで、値は返さない。

    Notes
    -----
    黒番は ``PolicyAgent``、白番は ``RandomAgent`` が担当する。
    """
    # 学習済みモデルを読み込んだPolicyAgent(黒番)とRandomAgent(白番)で対戦
    env = GomokuEnv(board_size=9)
    # 黒番となる学習済みエージェントを初期化
    black_agent = PolicyAgent(board_size=9, network_type=network_type)
    black_agent.load_model(policy_path)

    # 白番は単純なランダムエージェント
    white_agent = RandomAgent()

    # ゲーム環境の初期化
    obs = env.reset()
    done = False

    while not done:
        # どちらの手番かで使用するエージェントを切り替える
        if env.current_player == 1:  # 黒
            action = black_agent.get_action(obs, env)
        else:  # 白
            action = white_agent.get_action(obs, env)

        # 行動を環境へ適用し盤面を更新
        obs, reward, done, info = env.step(action)
        env.render()  # 盤面を表示

    # ゲーム終了後に勝者を判定
    winner = info["winner"]
    if winner == 1:
        print("黒番(PolicyAgent)の勝ち！")
    elif winner == 2:
        print("白番(RandomAgent)の勝ち！")
    else:
        print("引き分け")

if __name__ == "__main__":
    play_game_with_trained_model(
        MODEL_DIR / "policy_agent_trained.pth", network_type="dense"
    )
