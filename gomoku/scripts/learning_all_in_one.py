"""ゲーム環境から学習ループまでを一つにまとめた実験用スクリプト。

五目並べ環境の実装と強化学習によるエージェント学習を
このファイルだけで確認できる。
"""

from pathlib import Path

# 可能な限りシンプルな構成とするため、詳細な処理は別モジュールに分割した
from .config_defaults import DEFAULT_CONFIG
from .learning_runner import train_q_vs_q, play_trained_match

# 学習済みモデルを保存するディレクトリ
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# ------------------------------------------------------------
# グラフ画像を保存するディレクトリ
#   実験結果の可視化を GUI 無し環境でも確認できるよう、
#   ここにまとめて保存する
# ------------------------------------------------------------

def main():
    """デフォルト設定で QAgent 同士を学習させる"""

    # 設定は ``config_defaults`` にまとめてあるものをそのまま使用
    config = DEFAULT_CONFIG

    # サンプルとして QAgent の自己対戦学習を実行
    train_q_vs_q(config, show_plot=False)

    return config


if __name__ == "__main__":
    config = main()
    # 学習済みモデルを読み込み 1 ゲームだけ対戦を可視化
    play_trained_match(config, pause_time=0.5)

