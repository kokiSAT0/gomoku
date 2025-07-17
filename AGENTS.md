# AGENTS 開発メモ

このリポジトリは Python で五目並べ AI を作成するための個人用プロジェクトです。
ヒューリスティック手法から強化学習モデルまでを実装し、最終的には強い友人を負かす AI を作ることを目標とします。

## 目的
- 五目並べのゲームロジックと学習環境の実装
- AI 同士の対戦およびプレイヤー対 AI の対戦機能の作成
- 機械学習を学びながら強化学習モデルを構築する

## 推奨環境
- Python 3.10 以上 (開発用コンテナでは 3.12 系を使用)
- 主要ライブラリ: numpy, torch, pygame, tqdm, matplotlib
- 仮想環境を利用すると依存関係を管理しやすい

### 仮想環境の例
```bash
python -m venv venv
source venv/bin/activate
pip install numpy torch pygame tqdm matplotlib
```

## 主要スクリプト
- `gomoku/scripts/play_with_pygame.py`: Pygame を用いた GUI 対戦プログラム
- `gomoku/scripts/play_vs_model.py`: 端末上でモデルと対戦するための簡易インタフェース
- `gomoku/scripts/learn_model.py` / `gomoku/scripts/learning_all_in_one.py`: 強化学習によるモデル学習
- `gomoku/scripts/evaluate_models.py`: 学習済みモデルの評価
- `gomoku/scripts/parallel_pg_train.py` , `gomoku/scripts/parallel_q_train.py`: 並列での学習実験用

## コーディング方針
- コメントや docstring は必ず日本語で記述すること
- 可読性を高めるために処理の意図や数式の意味などをなるべく詳細にコメントする
- コードフォーマットは基本的に PEP8 に合わせつつ読みやすさを優先

## その他
- 学習済みモデルの `.pth` ファイルは `models/` フォルダにまとめて配置する
  必要に応じて各エージェントクラスから読み込む
- 新規機能やルール変更などがあればこの AGENTS.md に追記して履歴を残す
- README.md も常に最新の実装内容を説明するよう、機能追加のたびに更新する
