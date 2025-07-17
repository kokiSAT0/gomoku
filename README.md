# 五目並べAI プロジェクト

本リポジトリは Python を用いた五目並べ(Gomoku) の AI 学習・対戦環境です。
ヒューリスティック手法から強化学習モデルまで実装しており、GUI での対戦も可能です。
初心者の方でも実行できるよう、基本的な使い方をまとめます。

## 環境準備

- Python 3.10 以上 (開発用コンテナでは 3.12 系を使用)
- 必要ライブラリ: numpy, torch, pygame, tqdm, matplotlib, seaborn

仮想環境を利用する場合の例を以下に示します。
```bash
python -m venv .venv
.venv\Scripts\activate  # source venv/bin/activate
# requirements.txt を用意しているので以下のコマンド1つで依存ライブラリを
# まとめてインストールできます
pip install -r requirements.txt
```

## ファイル構成

- `gomoku_env.py` : ゲームルールおよび強化学習用環境
- `agents.py` : ランダム・ヒューリスティック・強化学習エージェント群
- `play_with_pygame.py` : Pygame を使った GUI 対戦プログラム
- `play_vs_model.py` : 学習済みモデルと端末上で対戦する簡易インタフェース
- `learn_model.py` : 自己対戦によるモデル学習スクリプト
- `evaluate_models.py` : 学習済みモデルの勝率評価
- `parallel_train.py` : 並列処理を用いた学習実験用サンプル
- `round_robin.py` : 複数エージェントの総当たり戦

`models/` フォルダには学習済みモデル(`*.pth`)をまとめて保存しています。

## 基本的な使い方

### 1. GUI で遊ぶ

```bash
python play_with_pygame.py
```

初期設定では学習済み `PolicyAgent` とヒューリスティックエージェントの対戦を
1 ゲームだけ可視化します。盤面ウィンドウが表示され、対局の様子を見ることができます。

### 2. 端末上で学習済みモデルと対戦する

```bash
python play_vs_model.py
```

黒番として学習済み `PolicyAgent` をロードし、白番のランダムエージェントと対戦します。
盤面はテキストで表示されます。

### 3. モデルを学習する

`learn_model.py` にはいくつかの自己対戦パターンが用意されています。例として
`PolicyAgent` 同士で学習する場合は以下のように実行します。

```bash
python learn_model.py
```

デフォルトでは 9x9 の盤面で 4000 エピソードの自己対戦を行い、
終了後に重みを `policy_agent_black.pth` 等へ保存します。
学習パラメータやエージェントの組み合わせはスクリプト内で変更してください。

### 4. モデルの評価

`evaluate_models.py` を用いると、学習済みモデルと任意のエージェントとの対戦を
複数回行い勝率を測定できます。

```bash
python evaluate_models.py
```

必要に応じて `policy_path` や `opponent_agent` を変更してください。

### 5. 並列学習や総当たり戦

より多くのデータを集めたい場合は `parallel_train.py` を参考に
マルチプロセスで学習を回すことができます。また、複数エージェントを比較したい
場合は `round_robin.py` の総当たり戦機能を使うと便利です。

### 6. 学習グラフの保存先

学習スクリプトを実行すると、報酬や勝率の推移を表すグラフが自動的に
`figures/` フォルダへ保存されます。GUI の無い環境では `show=False` を
指定することで表示を省略し、画像のみ出力することができます。

## 注意点

- Pygame を使用するスクリプトはディスプレイ出力が必要です。
- モデル学習には時間がかかるため、GPU 環境があると快適です。
- コード内のコメントはすべて日本語で記述してあります。

以上が基本的な使用方法です。詳しい処理やパラメータの意味などは各スクリプトの
コメントを参照してください。
