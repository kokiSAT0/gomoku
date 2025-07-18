# 五目並べAI プロジェクト

本リポジトリは Python を用いた五目並べ(Gomoku) の AI 学習・対戦環境です。
ヒューリスティック手法から強化学習モデルまで実装しており、GUI での対戦も可能です。
初心者の方でも実行できるよう、基本的な使い方をまとめます。

## 実装済み機能

- `gomoku/core` で盤面管理と勝敗判定を行う環境 `GomokuEnv` を実装
- `gomoku/ai` には以下のエージェントが存在
  - ランダムエージェント
  - 即勝ち・ブロック優先などのヒューリスティックエージェント
  - 方策勾配を用いた `PolicyAgent`
    - 畳み込み型の `ConvPolicyNet` と全結合型の `PolicyNet` を選択可能
  - DQN を用いた `QAgent`
- `gomoku/scripts` には GUI 対戦やモデル学習、評価、並列学習などのスクリプトを用意
- 学習曲線などの画像は `figures/` フォルダに出力
- 学習済みモデルは容量削減のためリポジトリに含まれていません。必要に応じて個別にダウンロードしてください。

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

## ディレクトリ構成

- `gomoku/core/` : ゲームロジックや補助関数を提供
- `gomoku/ai/` : さまざまなエージェント実装を格納
- `gomoku/scripts/` : 学習・対戦・評価用スクリプト

学習済みモデルは容量の都合上リポジトリに含めていません。必要に応じて個別に入手してください。

## 基本的な使い方

### 1. GUI で遊ぶ

```bash
python -m gomoku.scripts.play_with_pygame
```

初期設定では学習済み `PolicyAgent` とヒューリスティックエージェントの対戦を
1 ゲームだけ可視化します。盤面ウィンドウが表示され、対局の様子を見ることができます。

### 2. 端末上で学習済みモデルと対戦する

```bash
python -m gomoku.scripts.play_vs_model
```

黒番として学習済み `PolicyAgent` をロードし、白番のランダムエージェントと対戦します。
盤面はテキストで表示されます。
既存のモデルが全結合ネットワークで学習されている場合は
`network_type="dense"` を指定してください。

### 3. モデルを学習する

`gomoku/scripts/learn_model.py` では自己対戦ループを `selfplay パッケージ` に分割し、
複数の組み合わせを手軽に試せるようにしています。
`PolicyAgent` 同士で学習する場合は以下のように実行します。

```bash
python -m gomoku.scripts.learn_model
```

デフォルトでは 9x9 の盤面で 4000 エピソードの自己対戦を行い、
終了後に重みを `policy_agent_black.pth` 等へ保存します。
学習パラメータやエージェントの組み合わせはスクリプト内で変更してください。

### 4. モデルの評価

`gomoku/scripts/evaluate_models.py` を用いると、学習済みモデルと任意のエージェントとの対戦を
複数回行い勝率を測定できます。

```bash
python -m gomoku.scripts.evaluate_models
```

必要に応じて `policy_path` や `opponent_agent` を変更してください。
全結合ネットワークで学習したモデルを評価する場合は
`network_type="dense"` を引数に指定します。
`QAgent` を評価したい場合は `--agent_type q --q_path <モデルファイル>` を
付与してください。

### 5. 並列学習や総当たり戦

より多くのデータを集めたい場合は `gomoku/scripts/parallel_pg_train.py` と `gomoku/scripts/parallel_q_train.py` を参考に
マルチプロセスで学習を回すことができます。また、複数エージェントを比較したい
場合は `gomoku/scripts/round_robin.py` の総当たり戦機能を使うと便利です。

### 6. 学習グラフの保存先

学習スクリプトを実行すると、報酬や勝率の推移を表すグラフが自動的に
`figures/` フォルダへ保存されます。GUI の無い環境では `show=False` を
指定することで表示を省略し、画像のみ出力することができます。

### 7. 複数の相手に対する学習

`train_policy_mixed.py` を実行すると、
ヒューリスティックエージェントと既存の `PolicyAgent` を
ランダムに相手として学習を進められます。

```bash
python -m gomoku.scripts.train_policy_mixed --episodes 2000
```

学習後はそれぞれの相手に対する勝率が表示され、
モデルは `models/policy_mixed.pth` に保存されます。

### 8. QAgent とヒューリスティックの段階的学習

`train_q_vs_heuristics.py` を使うと、複数のヒューリスティックエージェントを
相手にフェーズを区切って `QAgent` を訓練できます。

```bash
python -m gomoku.scripts.train_q_vs_heuristics --episodes 500 --device cuda
```

盤面サイズを小さくして手早く試す場合は次のように指定します。

```bash
python -m gomoku.scripts.train_q_vs_heuristics --board-size 5 --episodes 100
```

より安定した結果を得るためにエピソード数を増やしたいときは以下のように実行します。

```bash
python -m gomoku.scripts.train_q_vs_heuristics --episodes 2000 --device cuda
```

学習は ``check_interval`` ごとに勝率を確認し、伸びがほぼ無くなった段階で
フェーズを途中終了します。 ``--interactive`` オプションを付けると、この
早期終了時に次のフェーズへ進むかを確認してくれます。最後に 1 戦だけテキス
ト表示で対局結果を確認できます。GPU を利用する場合は ``--device cuda`` を
指定します。
なお評価間隔は ``--check-interval`` オプションで変更可能で、
デフォルトは 100 エピソードごととなっています。

## 注意点

- Pygame を使用するスクリプトはディスプレイ出力が必要です。
- モデル学習には時間がかかるため、GPU 環境があると快適です。
- コード内のコメントはすべて日本語で記述してあります。

以上が基本的な使用方法です。詳しい処理やパラメータの意味などは各スクリプトの
コメントを参照してください。
