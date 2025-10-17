# app.py
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import os, traceback
import numpy as np
from joblib import load
from scipy.signal import find_peaks

MODEL_PATH = os.getenv("MODEL_PATH", "/models/my_model.joblib")

CLASS_LABELS = ["0", "10", "15", "20", "25"]

# デフォのピーク検出パラメータ
DEFAULT_HEIGHT = 0.0
DEFAULT_PROMINENCE = 0.01

app = FastAPI(title="Stroke Predictor API (sklearn, stroke input)", version="2.0.0")

# ====== 入出力スキーマ ======

class StrokePoint(BaseModel):
    x: float
    y: float
    pressure: float

class PredictRequest(BaseModel):
    # どちらか一方を指定：points（推奨） or features（後方互換）
    points: Optional[List[StrokePoint]] = Field(None, description="list of {x,y,pressure}")
    features: Optional[List[float]] = Field(None, description="[peak_count, interval_1, interval_1_5, interval_2, interval_2_5]")
    # オプション: ピーク検出パラメータ（pointsを使う時のみ有効）
    height: float = DEFAULT_HEIGHT
    prominence: float = DEFAULT_PROMINENCE

class PredictResponse(BaseModel):
    label: str
    features: List[float]  # 予測に使った最終特徴（可視化/検証用）

class ProbaResponse(BaseModel):
    labels: List[str]
    probs: List[float]
    features: List[float]

# ====== モデルロード ======

clf = None
feature_names = None   # 学習時の列名（あれば）
classes_out = None
n_features_in = None

@app.on_event("startup")
def on_startup():
    global clf, feature_names, classes_out, n_features_in
    clf = load(MODEL_PATH)
    feature_names = list(getattr(clf, "feature_names_in_", [])) or None
    classes_out = [str(c) for c in getattr(clf, "classes_", [])] or None
    n_features_in = int(getattr(clf, "n_features_in_", 0)) or None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "n_features_in": n_features_in,
        "feature_names_in": feature_names,
        "classes": classes_out or CLASS_LABELS
    }

@app.get("/model_info")
def model_info():
    return {
        "expected_n_features": n_features_in,
        "expected_feature_names": feature_names or ["peak_count","interval_1","interval_1_5","interval_2","interval_2_5"],
        "classes": classes_out or CLASS_LABELS
    }

# ====== 特徴抽出 ======

def featurize_from_stroke(points: List[StrokePoint], height: float, prominence: float) -> List[float]:
    """
    {x,y,pressure} の配列から特徴5つを返す:
      [peak_count, interval_1, interval_1_5, interval_2, interval_2_5]
    """
    if len(points) < 3:
        # 短すぎる → 全ゼロで返す
        return [0, 0, 0, 0, 0]

    xs = np.array([p.x for p in points], dtype=float)
    ps = np.array([p.pressure for p in points], dtype=float)

    # ピーク検出（scipy）
    # height は下限、prominence は突出度。
    peak_idx, _ = find_peaks(ps, height=height, prominence=prominence)
    peak_count = int(len(peak_idx))

    # ピーク間の x 距離でビン分け
    interval_1 = interval_1_5 = interval_2 = interval_2_5 = 0
    if peak_count > 1:
        dx = np.diff(xs[peak_idx])  # 連続ピーク間の x 差
        for d in dx:
            # 元コードの閾値（半開区間/閉区間の扱いも踏襲）
            if 4.8975 <= d < 8.1625:
                interval_1 += 1
            elif 8.1625 <= d < 11.4275:
                interval_1_5 += 1
            elif 11.4275 <= d <= 14.6925:
                interval_2 += 1
            elif 14.6925 <= d <= 17.9575:
                interval_2_5 += 1

    return [peak_count, interval_1, interval_1_5, interval_2, interval_2_5]

def _build_X_from_features(feats: List[float]):
    # モデルが期待する本数に合わせる
    if n_features_in is not None and len(feats) != n_features_in:
        raise HTTPException(
            status_code=400,
            detail=f"feature length mismatch: got {len(feats)}, expected {n_features_in}"
        )
    # 学習時に列名があるなら DataFrame で順序固定（Warning回避）
    try:
        import pandas as pd
        cols = feature_names
        if cols and len(cols) == len(feats):
            return pd.DataFrame([feats], columns=cols)
    except Exception:
        pass
    return np.array(feats, dtype=float).reshape(1, -1)

def _get_features_from_request(req: PredictRequest) -> List[float]:
    if req.points and len(req.points) > 0:
        return featurize_from_stroke(req.points, req.height, req.prominence)
    if req.features and len(req.features) > 0:
        return [float(v) for v in req.features]
    raise HTTPException(status_code=400, detail="provide either non-empty 'points' or 'features'")

# ====== エンドポイント ======

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        feats = _get_features_from_request(req)
        X = _build_X_from_features(feats)
        y = clf.predict(X)[0]
        return PredictResponse(label=str(y), features=feats)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"predict failed: {repr(e)} | {traceback.format_exc(limit=1)}")

@app.post("/proba", response_model=ProbaResponse)
def proba(req: PredictRequest):
    if not hasattr(clf, "predict_proba"):
        raise HTTPException(status_code=400, detail="model has no predict_proba")
    try:
        feats = _get_features_from_request(req)
        X = _build_X_from_features(feats)
        probs = clf.predict_proba(X)[0].tolist()
        labels = [str(c) for c in getattr(clf, "classes_", range(len(probs)))]
        if len(labels) != len(probs):
            raise HTTPException(status_code=500, detail="probability length mismatch")
        return ProbaResponse(labels=labels, probs=probs, features=feats)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"proba failed: {repr(e)} | {traceback.format_exc(limit=1)}")

# デバッグ用: 特徴だけ返す
@app.post("/featurize")
def featurize(req: PredictRequest):
    feats = _get_features_from_request(req)
    return {
        "features": {
            "peak_count": feats[0],
            "interval_1": feats[1],
            "interval_1_5": feats[2],
            "interval_2": feats[3],
            "interval_2_5": feats[4],
        }
    }
