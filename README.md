# 溝間隔の違いによる筆圧変化を活用したシート埋め込み型 ID 認識手法

FastAPI ベースの推論 API で、手書きストロークから ID（溝間隔パターン）を推定する研究プロトタイプです。圧力センサ付きシートから取得したストローク系列を、ピーク検出と区間統計で特徴量化し、scikit-learn で学習済みのモデル (`models/my_model.joblib`) に入力します。

- **研究背景**: 溝間隔の違いによる筆圧変化を利用してシート埋め込み型 ID を判別する手法の実装検証用
- **主な役割**: ストローク系列（`x`, `y`, `pressure`）から ID ラベル (`0`, `10`, `15`, `20`, `25`) を推定
- **実装技術**: FastAPI, scikit-learn, NumPy, SciPy（ピーク検出）

---

## 構成概要

```
.
├── app.py               # 推論 API 本体（FastAPI）
├── create_model.ipynb   # 特徴抽出・学習ノートブック
├── models/
│   └── my_model.joblib  # 学習済みモデル
├── Dockerfile           # Alpine ベースのコンテナ定義
├── docker-compose.yml   # 本番/PoC 向けの Compose 設定
├── requirements.txt     # Python 依存関係
└── Makefile             # ビルド・起動・確認コマンド
```

---

## セットアップ

### 1. Docker / Docker Compose

```bash
# イメージビルド
make build

# 背景起動（ポート 8000）
make up

# ログ表示
make logs

# 停止・クリーンアップ
make down        # 停止のみ
make clean       # イメージ・ボリュームも削除
```

---

## API エンドポイント

| Method | Path         | 説明                                               |
|--------|--------------|----------------------------------------------------|
| GET    | `/health`    | モデルロード状態やクラス一覧の確認                 |
| GET    | `/model_info`| 期待する特徴量数・列名などのメタ情報              |
| POST   | `/predict`   | 単一ストロークから ID ラベルを推定 (`PredictResponse`) |
| POST   | `/proba`     | ラベルごとの確率を取得 (`ProbaResponse`)           |
| POST   | `/featurize` | 特徴量抽出のみ（ピーク数や区間ヒストグラム）      |

### リクエスト例（`/predict`）

```bash
curl -s http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "points": [
          {"x": 0, "y": 0, "pressure": 0.01},
          {"x": 5, "y": 0, "pressure": 0.20},
          {"x": 10, "y": 0, "pressure": 0.02}
        ],
        "height": 0.0,
        "prominence": 0.01
      }'
```

`points` の代わりに、事前抽出した特徴量ベクトル `features` を直接渡すことも可能です。

---

## 特徴量設計の概要

1. **ピーク検出**: SciPy の `find_peaks` で筆圧の局所ピークを抽出します。高さ (`height`)、突出度 (`prominence`) は API パラメータで調整可能。
2. **区間ヒストグラム**: 連続するピーク間の x 座標差を計算し、研究で定義された 4 つの区間（約 4.9〜18 mm）にカウント分布化。
3. **最終特徴ベクトル**: `[peak_count, interval_1, interval_1_5, interval_2, interval_2_5]` の 5 次元ベクトルを scikit-learn モデルに渡します。

処理の詳細は `app.py` の `featurize_from_stroke` 関数に実装されています。

---

## モデルの更新

- 学習・評価手順は `create_model.ipynb` にまとめています。
- 新しいモデルを作成したら `models/` 配下に配置し、環境変数 `MODEL_PATH` で参照するパスを指定してください。
- API 起動時に `joblib.load(MODEL_PATH)` で自動ロードされます。

---

## ライセンス / 引用
後日記載