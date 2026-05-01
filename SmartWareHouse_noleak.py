# ============================================================
# SmartWarehouse 최종 통합 코드 (look-ahead 제거본)
#
# SmartWareHouse_fixed.py 의 6가지 수정에 추가로
# 아래 4가지 "미수정 항목"을 모두 반영한 버전.
#
# [추가 수정]
# Fix 7. handle_missing()에서 bfill() 제거
#         → 미래 타임슬롯 값으로 결측치를 채우는 look-ahead 방지
#           (bfill 대신 ffill + 전체 median 대체)
# Fix 8. add_interaction_features()의 시나리오 전체 집계를
#         expanding 누적 통계로 교체
#         → timeslot t 행이 t+1 이후 피처 값을 포함하는 look-ahead 방지
#           groupby.agg(전체) → groupby.expanding() 누적값 사용
# Fix 9. add_interaction_features()의 rank(pct=True)를
#         expanding rank로 교체
#         → 동일한 look-ahead 방지
# Fix 10. Optuna HPO 평가를 sample_folds[0] 단일 fold →
#          3-fold 평균 MAE 기준으로 변경
#          → 특정 split에 과적합된 하이퍼파라미터 선택 방지
# Fix 11. ENABLE_TIMESLOT_ALPHA_SEARCH = False,
#          ENABLE_GLOBAL_CALIBRATION_SEARCH = False
#          → 단일 OOF 세트에 반복 후처리 최적화하는 과적합 위험 제거
#
# [상속된 수정 — SmartWareHouse_fixed.py 와 동일]
# Fix 1~6: Optuna AttributeError / TPE seed / CUDA 결정론 /
#           라이브러리 버전 출력 / 상대 경로 / 카테고리 일관 인코딩
# ============================================================

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# [Fix 1] optuna.logging 호출을 try 블록 안으로 이동
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None
    OPTUNA_AVAILABLE = False

from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

# [Fix 3] GPU 결정론 설정
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════
SEED = 42
N_FOLDS = 5
USE_LOG_TARGET = True

USE_OPTUNA_HPO = True
OPTUNA_TRIALS = 20
OPTUNA_SAMPLE_FRAC = 0.4

USE_STAGE2 = True
BLEND_ALPHA_STEP = 0.01
CLIP_Q = 0.995

# [Fix 11] OOF 기반 후처리 과적합 방지: 두 옵션 모두 비활성화
ENABLE_TIMESLOT_ALPHA_SEARCH    = False
ENABLE_GLOBAL_CALIBRATION_SEARCH = False
TIMESLOT_ALPHA_STEP              = 0.02   # 비활성화되어 있으나 변수 유지
TIMESLOT_ALPHA_SMOOTH_TO_GLOBAL  = 0.20

P_GRID    = [1.0, 1.5, 2.0, 3.0, 4.0]
LGB_SEEDS = [SEED, SEED + 100]

USE_AUTOENCODER    = True
USE_TARGET_ENCODING = True
USE_SAMPLE_WEIGHT  = True
USE_DOMAIN_CLIP    = True
USE_MISSING_MASK   = True

SW_HIGH_Q        = 0.90
SW_HIGH_MULT     = 1.25
SW_TS_EDGE_MULT  = 1.10

AE_LATENT_DIM  = 32
AE_HIDDEN_DIM  = 256
AE_EPOCHS_MAX  = 30
AE_ES_PATIENCE = 5
AE_BATCH_SIZE  = 4096
AE_LR          = 1e-3
AE_WEIGHT_DECAY = 3e-5
AE_DROPOUT     = 0.0
AE_DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TE_SMOOTHING = 20

RATIO_KEYWORDS = ["ratio", "utilization", "efficiency", "pct", "percent", "share"]
NONNEG_KEYWORDS = [
    "count", "cnt", "num", "minutes", "min", "hour", "distance",
    "delay", "inflow", "order", "robot", "battery", "density",
    "area", "weight", "pressure", "risk", "wait", "trip",
]

TARGET   = "avg_delay_minutes_next_30m"
ID_COLS  = ["ID", "layout_id", "scenario_id"]

PRED_LAG_COLS = [
    "pred_lag1", "pred_lag2", "pred_lag3",
    "pred_diff1", "pred_diff2",
    "pred_roll3_mean", "pred_roll5_mean",
    "pred_ewm3", "pred_lag1_log",
]


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════
def elapsed(start):
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60:02d}s"


def section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def to_train_target(y):
    return np.log1p(y) if USE_LOG_TARGET else y


def from_train_pred(p):
    return np.expm1(p) if USE_LOG_TARGET else p


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def _powerset_models():
    return [
        ["lgb"], ["xgb"], ["cat"],
        ["lgb", "xgb"], ["lgb", "cat"], ["xgb", "cat"],
        ["lgb", "xgb", "cat"],
    ]


def ensemble_pred(oof_by, maes_by, models, p):
    w  = {m: 1.0 / (maes_by[m] ** p) for m in models}
    ws = sum(w.values())
    return sum(w[m] * oof_by[m] for m in models) / ws


def apply_scale_bias_clip(pred, scale=1.0, bias=0.0, clip_q=None):
    pred  = np.asarray(pred, dtype=np.float64) * scale + bias
    pred  = np.clip(pred, 0.0, None)
    upper = None
    if clip_q is not None:
        upper = float(np.percentile(pred, clip_q))
        pred  = np.clip(pred, 0.0, upper)
    return pred, upper


# [Fix 6] train 기준 카테고리 일관 인코딩
def encode_objects_from_train(train_df, test_df, exclude=("ID",)):
    common_cols = train_df.columns.intersection(test_df.columns)
    for c in common_cols:
        if c in exclude:
            continue
        tr_is_obj = not pd.api.types.is_numeric_dtype(train_df[c])
        te_is_obj = not pd.api.types.is_numeric_dtype(test_df[c])
        if tr_is_obj or te_is_obj:
            codes, uniques = pd.factorize(train_df[c].astype(str).fillna("missing"))
            train_df[c] = codes.astype(np.int32)
            mapping = {v: i for i, v in enumerate(uniques)}
            test_df[c] = (
                test_df[c].astype(str).fillna("missing")
                .map(mapping).fillna(-1).astype(np.int32)
            )
    for c in train_df.columns:
        if c in exclude or c == TARGET:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[c]):
            train_df[c] = pd.factorize(train_df[c].astype(str).fillna("missing"))[0].astype(np.int32)
    for c in test_df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(test_df[c]):
            test_df[c] = pd.factorize(test_df[c].astype(str).fillna("missing"))[0].astype(np.int32)
    return train_df, test_df


def assert_numeric_features(df, feature_cols, name):
    bad = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"{name}에 숫자가 아닌 feature가 있습니다: {bad[:20]}")


def build_sample_weight(df):
    w = np.ones(len(df), dtype=np.float64)
    if not USE_SAMPLE_WEIGHT or TARGET not in df.columns:
        return w
    y     = df[TARGET].to_numpy(dtype=np.float64)
    q_thr = float(np.quantile(y, SW_HIGH_Q))
    w[y >= q_thr] *= SW_HIGH_MULT
    if "timeslot" in df.columns:
        ts = df["timeslot"].to_numpy(dtype=np.int64)
        w[(ts <= 2) | (ts >= 22)] *= SW_TS_EDGE_MULT
    return w


# ════════════════════════════════════════════════════════════════
# AutoEncoder
# ════════════════════════════════════════════════════════════════
class TabularAutoEncoder(nn.Module):
    def __init__(self, n_in, hidden, latent, dropout=0.0):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(hidden, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(hidden, n_in),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def ae_prepare_matrix(df, cols, medians):
    X = df[cols].to_numpy(dtype=np.float64)
    for j, col in enumerate(cols):
        m = float(medians[col])
        v = X[:, j]
        v[~np.isfinite(v)] = m
        v[np.isnan(v)]     = m
    return X


def ae_train_fold(X_tr, X_va, device, seed):
    torch.manual_seed(seed)
    n_in  = X_tr.shape[1]
    model = TabularAutoEncoder(n_in, AE_HIDDEN_DIM, AE_LATENT_DIM, AE_DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
    scaler = StandardScaler()
    tr_t   = torch.from_numpy(scaler.fit_transform(X_tr)).float().to(device)
    va_t   = torch.from_numpy(scaler.transform(X_va)).float().to(device)

    best_val, best_state, no_improve = float("inf"), None, 0
    for _ in range(AE_EPOCHS_MAX):
        model.train()
        perm = torch.randperm(tr_t.shape[0], device=device)
        for i in range(0, len(perm), AE_BATCH_SIZE):
            xb = tr_t[perm[i: i + AE_BATCH_SIZE]]
            opt.zero_grad(set_to_none=True)
            xh, _ = model(xb)
            nn.functional.mse_loss(xh, xb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(model(va_t)[0], va_t).item())
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= AE_ES_PATIENCE:
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        z_va = model(va_t)[1].cpu().numpy()
    return scaler, {k: v.cpu().clone() for k, v in model.state_dict().items()}, z_va


def ae_encode(X_scaled, state_dict, device):
    n_in  = X_scaled.shape[1]
    model = TabularAutoEncoder(n_in, AE_HIDDEN_DIM, AE_LATENT_DIM, AE_DROPOUT).to(device)
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    model.eval()
    xt  = torch.from_numpy(X_scaled).float().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), AE_BATCH_SIZE):
            out.append(model(xt[i: i + AE_BATCH_SIZE])[1].cpu().numpy())
    return np.vstack(out)


# ════════════════════════════════════════════════════════════════
# Feature Engineering
# ════════════════════════════════════════════════════════════════
SCENARIO_AGG_COLS = [
    "congestion_score", "order_inflow_15m", "battery_mean", "pack_utilization",
    "avg_trip_distance", "low_battery_ratio", "max_zone_density", "sku_concentration",
    "robot_idle", "outbound_truck_wait_min", "order_per_station", "robot_efficiency",
    "order_pressure", "risk_index", "battery_risk", "battery_cv",
    "zone_dispersion", "avg_items_per_order", "cold_chain_ratio",
    "manual_override_ratio", "robot_total", "battery_std",
    "bottle_neck", "trip_congestion", "pack_order_ratio", "robot_per_station",
]

TS_COLS = [
    "order_inflow_15m", "robot_active", "robot_total",
    "pack_utilization", "congestion_score", "avg_trip_distance",
    "low_battery_ratio", "outbound_truck_wait_min",
    "order_per_station", "robot_efficiency",
]

EWM_CORE_COLS = ["order_inflow_15m", "congestion_score", "pack_utilization"]
ROLL_WINDOWS  = (3, 5, 10)


def add_missing_indicators(df):
    miss = [c for c in df.columns if c not in ID_COLS + [TARGET] and df[c].isnull().any()]
    for c in miss:
        df[f"{c}_isna"] = df[c].isnull().astype(np.int8)
    return df


# [Fix 7] bfill() 제거: 미래 값으로 결측치를 채우는 look-ahead 방지
def handle_missing(df):
    cols = [c for c in df.columns if df[c].isnull().any() and c not in ID_COLS + [TARGET]]
    if cols:
        df[cols] = df.groupby("scenario_id")[cols].ffill()
        # bfill 제거 → 남은 결측치는 전체 median으로 대체
        df[cols] = df[cols].fillna(df[cols].median())
    return df


def clip_domain(df):
    for c in df.columns:
        if c in ID_COLS + [TARGET] or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        lc = c.lower()
        if any(k in lc for k in RATIO_KEYWORDS):
            df[c] = df[c].clip(0.0, 1.0)
        elif any(k in lc for k in NONNEG_KEYWORDS):
            df[c] = df[c].clip(lower=0.0)
    return df


def add_basic_features(df):
    df["timeslot"]         = df.groupby("scenario_id").cumcount()
    df["robot_efficiency"] = df["robot_active"] / (df["robot_total"] + 1e-6)
    df["robot_density"]    = df["robot_total"] / (df["floor_area_sqm"] + 1e-6)
    df["order_per_station"] = df["order_inflow_15m"] / (df["pack_station_count"] + 1e-6)
    df["robot_per_station"] = df["robot_active"] / (df["pack_station_count"] + 1e-6)
    df["cumulative_orders"] = df.groupby("scenario_id")["order_inflow_15m"].cumsum()
    df["order_pressure"]    = df["cumulative_orders"] / (df["pack_station_count"] + 1e-6)

    if "congestion_score" in df.columns:
        df["risk_index"] = df["congestion_score"] * (1 - df["robot_efficiency"])
        df["bottle_neck"] = df["order_per_station"] * df["congestion_score"]
    if "low_battery_ratio" in df.columns:
        df["battery_risk"] = df["low_battery_ratio"] * df["robot_total"]
    if "battery_mean" in df.columns and "battery_std" in df.columns:
        df["battery_cv"] = df["battery_std"] / (df["battery_mean"] + 1e-6)
    if "avg_trip_distance" in df.columns:
        df["trip_per_robot"]  = df["avg_trip_distance"] / (df["robot_active"] + 1e-6)
        df["trip_congestion"] = df["avg_trip_distance"] * df.get("congestion_score", 0)
    if "pack_utilization" in df.columns:
        df["pack_order_ratio"] = df["pack_utilization"] / (df["order_per_station"] + 1e-6)
    return df


def add_timeseries_features(df):
    scen = df["scenario_id"]
    for col in TS_COLS:
        if col not in df.columns:
            continue
        g = df.groupby("scenario_id")[col]
        for lag_n in (1, 2, 3):
            df[f"{col}_lag{lag_n}"] = g.shift(lag_n)
        s1, s2, s3 = g.shift(1), g.shift(2), g.shift(3)
        df[f"{col}_diff1"] = s1 - s2
        df[f"{col}_diff2"] = s2 - s3
        for w in ROLL_WINDOWS:
            roll = s1.groupby(scen).rolling(window=w, min_periods=1)
            df[f"{col}_roll{w}_mean"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{col}_roll{w}_std"]  = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        if col in EWM_CORE_COLS:
            df[f"{col}_ewm_mean"] = (
                s1.groupby(scen).ewm(alpha=0.3, adjust=False)
                .mean().reset_index(level=0, drop=True)
            )

    lag_cols = [c for c in df.columns if ("_lag" in c or "_diff" in c) and c not in ID_COLS]
    if lag_cols:
        df[lag_cols] = df.groupby("scenario_id")[lag_cols].ffill()
        for c in lag_cols:
            base = c.split("_lag")[0].split("_diff")[0]
            if base in df.columns:
                df[c] = df[c].fillna(df.groupby("scenario_id")[base].transform("mean"))
        df[lag_cols] = df[lag_cols].fillna(df[lag_cols].median())
    return df


def add_interaction_features(df):
    pairs = [
        ("congestion_score", "pack_utilization", "cong_x_pack"),
        ("congestion_score", "avg_trip_distance", "cong_x_trip"),
        ("order_per_station", "congestion_score", "ops_x_cong"),
        ("order_per_station", "pack_utilization", "ops_x_pack"),
    ]
    for a, b, name in pairs:
        if a in df.columns and b in df.columns:
            df[name] = df[a] * df[b]
    if "low_battery_ratio" in df.columns and "robot_efficiency" in df.columns:
        df["lowbat_x_eff"] = df["low_battery_ratio"] * (1 - df["robot_efficiency"])

    if "timeslot" in df.columns:
        for col in ["congestion_score", "pack_utilization", "order_per_station", "low_battery_ratio"]:
            if col in df.columns:
                df[f"ts_x_{col}"] = df["timeslot"] * df[col]

    # [Fix 8] 시나리오 전체 집계 → expanding 누적 통계로 교체 (look-ahead 제거)
    # 원본: groupby.agg(["mean","max","min","std"]) → 전체 25 타임슬롯 사용
    # 수정: groupby.expanding() → timeslot t까지의 누적값만 사용
    present = [c for c in SCENARIO_AGG_COLS if c in df.columns]
    if present:
        for col in present:
            g = df.groupby("scenario_id")[col]
            df[f"{col}_scen_mean"] = g.expanding().mean().reset_index(level=0, drop=True)
            df[f"{col}_scen_max"]  = g.expanding().max().reset_index(level=0, drop=True)
            df[f"{col}_scen_min"]  = g.expanding().min().reset_index(level=0, drop=True)
            df[f"{col}_scen_std"]  = (
                g.expanding().std().reset_index(level=0, drop=True).fillna(0.0)
            )

    scen_pairs = [
        ("congestion_score_scen_mean", "pack_utilization_scen_mean", "cong_pack_interaction"),
        ("avg_trip_distance_scen_mean", "congestion_score_scen_mean", "trip_cong_interaction"),
        ("pack_utilization_scen_mean",  "avg_trip_distance_scen_mean", "pack_trip_interaction"),
        ("low_battery_ratio_scen_mean", "congestion_score_scen_mean", "battery_cong_interaction"),
        ("max_zone_density_scen_mean",  "congestion_score_scen_mean", "density_cong_interaction"),
    ]
    for a, b, name in scen_pairs:
        if a in df.columns and b in df.columns:
            df[name] = df[a] * df[b]
    if "low_battery_ratio_scen_mean" in df.columns and "robot_efficiency_scen_mean" in df.columns:
        df["battery_efficiency_risk"] = (
            df["low_battery_ratio_scen_mean"] * (1 - df["robot_efficiency_scen_mean"])
        )

    for col in ["congestion_score", "order_per_station", "pack_utilization", "avg_trip_distance"]:
        sm = f"{col}_scen_mean"
        if col in df.columns and sm in df.columns:
            df[f"{col}_rel_to_scen"] = df[col] / (df[sm] + 1e-6)

    # [Fix 9] 전체 시나리오 분포 기반 rank → expanding rank로 교체 (look-ahead 제거)
    # 원본: groupby.rank(pct=True) → 전체 25 타임슬롯 분포 기준 백분위
    # 수정: expanding().rank(pct=True) → timeslot t까지의 분포 기준 백분위
    for col in ["congestion_score", "order_per_station", "pack_utilization"]:
        if col in df.columns:
            df[f"{col}_scen_rank"] = df.groupby("scenario_id")[col].transform(
                lambda x: x.expanding().rank(pct=True)
            )
    return df


def add_extra_features(df):
    gsz  = df.groupby("scenario_id")["scenario_id"].transform("size")
    gpos = df.groupby("scenario_id").cumcount()
    df["row_frac_in_scen"] = (gpos + 1) / (gsz + 1e-6)
    if "order_inflow_15m" in df.columns:
        df["sqrt_order_inflow"] = np.sqrt(np.maximum(df["order_inflow_15m"].astype(np.float64), 0.0))
    if "robot_total" in df.columns:
        df["sqrt_robot_total"] = np.sqrt(np.maximum(df["robot_total"].astype(np.float64), 0.0))
    if "timeslot" in df.columns:
        ang = 2 * np.pi * df["timeslot"].astype(np.float64) / 24.0
        df["ts_sin"] = np.sin(ang)
        df["ts_cos"] = np.cos(ang)
    return df


# [Fix 6 + 8] preprocess_all: 독립 factorize 제거 (encode_objects_from_train이 담당)
def preprocess_all(df, layout_df):
    df = df.merge(layout_df, on="layout_id", how="left")
    df = df.sort_values(["scenario_id", "ID"]).reset_index(drop=True)
    if USE_MISSING_MASK:
        df = add_missing_indicators(df)
    df = handle_missing(df)        # [Fix 7] bfill 없음
    df = add_basic_features(df)
    df = add_timeseries_features(df)
    df = add_interaction_features(df)  # [Fix 8,9] expanding 통계/rank
    df = add_extra_features(df)
    if USE_DOMAIN_CLIP:
        df = clip_domain(df)
    return df


# ════════════════════════════════════════════════════════════════
# Pred-lag features (Stage 2용)
# ════════════════════════════════════════════════════════════════
def add_pred_lag_features(df, pred_vec, gm):
    df    = df.sort_values(["scenario_id", "ID"]).reset_index(drop=True)
    p     = np.asarray(pred_vec, dtype=np.float64)
    df["_pred"] = p
    g     = df.groupby("scenario_id")["_pred"]
    s1, s2, s3 = g.shift(1), g.shift(2), g.shift(3)
    df["pred_lag1"]      = s1
    df["pred_lag2"]      = s2
    df["pred_lag3"]      = s3
    df["pred_diff1"]     = (s1 - s2).fillna(0.0)
    df["pred_diff2"]     = (s2 - s3).fillna(0.0)
    df["pred_roll3_mean"] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["pred_roll5_mean"] = g.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["pred_ewm3"]       = g.transform(lambda x: x.shift(1).ewm(alpha=0.3, adjust=False).mean())
    df["pred_lag1_log"]   = np.log1p(np.maximum(df["pred_lag1"].fillna(gm).astype(np.float64), 0.0))
    for c in ["pred_lag1", "pred_lag2", "pred_lag3", "pred_roll3_mean", "pred_roll5_mean", "pred_ewm3"]:
        df[c] = df[c].fillna(gm)
    df.drop(columns=["_pred"], inplace=True)
    return df


# ════════════════════════════════════════════════════════════════
# Target Encoding
# ════════════════════════════════════════════════════════════════
def apply_target_encoding(train_df, test_df, kf, groups, global_mean):
    te_cols  = [c for c in ["layout_id", "timeslot", "layout_type"] if c in train_df.columns]
    te_pairs = [(a, b) for i, a in enumerate(te_cols) for b in te_cols[i + 1:]]

    def _te(col_name, tr_key, te_key):
        te_col = f"{col_name}_te"
        train_df[te_col] = np.nan
        for tr_idx, va_idx in kf.split(train_df, train_df[TARGET], groups=groups):
            tr     = train_df.iloc[tr_idx]
            st     = tr.groupby(tr_key.iloc[tr_idx])[TARGET].agg(["mean", "count"])
            smooth = (st["count"] * st["mean"] + TE_SMOOTHING * global_mean) / (st["count"] + TE_SMOOTHING)
            train_df.loc[train_df.index[va_idx], te_col] = tr_key.iloc[va_idx].map(smooth).fillna(global_mean)
        st_full     = train_df.groupby(tr_key)[TARGET].agg(["mean", "count"])
        smooth_full = (st_full["count"] * st_full["mean"] + TE_SMOOTHING * global_mean) / (st_full["count"] + TE_SMOOTHING)
        test_df[te_col] = te_key.map(smooth_full).fillna(global_mean)

    for col in te_cols:
        _te(col, train_df[col], test_df[col])
    for a, b in te_pairs:
        name = f"{a}_X_{b}"
        _te(name,
            train_df[a].astype(str) + "_" + train_df[b].astype(str),
            test_df[a].astype(str)  + "_" + test_df[b].astype(str))


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    T0 = time.time()

    # [Fix 4] 라이브러리 버전 기재
    print("=== 개발 환경 및 라이브러리 버전 ===")
    print(f"  Python    : {sys.version}")
    print(f"  OS        : {sys.platform}")
    print(f"  pandas    : {pd.__version__}")
    print(f"  numpy     : {np.__version__}")
    print(f"  lightgbm  : {lgb.__version__}")
    print(f"  xgboost   : {xgb.__version__}")
    print(f"  catboost  : {cb.__version__}")
    print(f"  torch     : {torch.__version__}")
    if OPTUNA_AVAILABLE:
        print(f"  optuna    : {optuna.__version__}")
    else:
        print("  optuna    : 미설치 (기본 파라미터 사용)")
    print("=" * 40)

    # ── 1. Data Load ──
    section("데이터 로드")
    # [Fix 5] 상대 경로
    data_dir = Path("data")
    if not (data_dir / "train.csv").exists():
        raise FileNotFoundError(
            "data/train.csv가 없습니다. 스크립트와 같은 폴더에 data/ 디렉터리를 배치하세요."
        )
    print(f"  data: {data_dir}")
    t0     = time.time()
    train  = pd.read_csv(data_dir / "train.csv")
    test   = pd.read_csv(data_dir / "test.csv")
    layout = pd.read_csv(data_dir / "layout_info.csv")
    print(f"  train {len(train):,} / test {len(test):,}  ({elapsed(t0)})")

    # ── 2. Preprocessing ──
    section("전처리")
    t0    = time.time()
    train = preprocess_all(train, layout)
    test  = preprocess_all(test,  layout)
    # [Fix 6] train 기준 일관 인코딩
    train, test = encode_objects_from_train(train, test, exclude=("ID",))
    print(f"  완료 ({elapsed(t0)})")

    # ── 3. Target Encoding ──
    if USE_TARGET_ENCODING:
        section("Target Encoding")
        t0          = time.time()
        global_mean = float(train[TARGET].mean())
        kf_te       = GroupKFold(n_splits=N_FOLDS)
        apply_target_encoding(train, test, kf_te, train["scenario_id"], global_mean)
        print(f"  완료 ({elapsed(t0)})")
    else:
        global_mean = float(train[TARGET].mean())

    # ── 4. Feature columns ──
    feature_cols_s1_base = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    ae_input_cols        = [c for c in feature_cols_s1_base if pd.api.types.is_numeric_dtype(train[c])]
    AE_COLS = [f"ae_z{i}" for i in range(AE_LATENT_DIM)] if USE_AUTOENCODER else []
    if USE_AUTOENCODER:
        for c in AE_COLS:
            train[c] = 0.0
            test[c]  = 0.0
        feature_cols_s1 = feature_cols_s1_base + AE_COLS
    else:
        feature_cols_s1 = list(feature_cols_s1_base)

    assert_numeric_features(train, feature_cols_s1_base, "train stage1 base")
    assert_numeric_features(test,  feature_cols_s1_base, "test stage1 base")
    print(f"  피처 수: {len(feature_cols_s1)} (base={len(feature_cols_s1_base)}, AE={len(AE_COLS)})")

    # ── 5. CV Setup ──
    y_all  = to_train_target(train[TARGET].values)
    y_raw  = train[TARGET].values
    groups = train["scenario_id"].values
    sw_all = build_sample_weight(train)

    _scen_mean = train.groupby("scenario_id")[TARGET].transform("mean")
    try:
        y_strat = pd.qcut(_scen_mean, q=10, labels=False, duplicates="drop")
        y_strat = pd.Series(y_strat).fillna(0).astype(np.int64).values
    except Exception:
        y_strat = np.zeros(len(train), dtype=np.int64)

    try:
        kf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        _  = list(kf.split(train, y_strat, groups=groups))
        print(f"  CV: StratifiedGroupKFold ({N_FOLDS}-fold)")
    except Exception:
        kf      = GroupKFold(n_splits=N_FOLDS)
        y_strat = y_all
        print(f"  CV: GroupKFold ({N_FOLDS}-fold)")

    # ── 6. Optuna HPO ──
    if USE_OPTUNA_HPO and OPTUNA_AVAILABLE:
        section(f"Optuna LightGBM 최적화 ({OPTUNA_TRIALS} trials)")
        rng          = np.random.default_rng(SEED)
        sample_scens = train["scenario_id"].unique()
        sample_scens = rng.choice(
            sample_scens, size=int(len(sample_scens) * OPTUNA_SAMPLE_FRAC), replace=False
        )
        sample_mask  = train["scenario_id"].isin(sample_scens)
        train_sample = train[sample_mask].reset_index(drop=True)
        y_sample     = to_train_target(train_sample[TARGET])
        sample_kf    = GroupKFold(n_splits=3)
        sample_folds = list(sample_kf.split(train_sample, y_sample, groups=train_sample["scenario_id"]))

        # [Fix 10] 3-fold 평균 MAE 기준으로 HPO (원본: fold[0] 단일 사용)
        def lgb_objective(trial):
            params = dict(
                objective="regression_l1",
                n_estimators=2000,
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
                max_depth=trial.suggest_int("max_depth", 5, 12),
                num_leaves=trial.suggest_int("num_leaves", 63, 2047),
                min_child_samples=trial.suggest_int("min_child_samples", 20, 100),
                subsample=trial.suggest_float("subsample", 0.6, 0.9),
                subsample_freq=1,
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 0.8),
                reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0),
                random_state=SEED, verbose=-1,
            )
            fold_maes = []
            for tr_idx, val_idx in sample_folds:  # 3개 fold 모두 사용
                X_tr      = train_sample.iloc[tr_idx][feature_cols_s1_base].replace([np.inf, -np.inf], np.nan)
                y_tr      = y_sample.iloc[tr_idx]
                X_val     = train_sample.iloc[val_idx][feature_cols_s1_base].replace([np.inf, -np.inf], np.nan)
                y_val_raw = train_sample.iloc[val_idx][TARGET]
                m = lgb.LGBMRegressor(**params)
                m.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_sample.iloc[val_idx])],
                    eval_metric="mae",
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
                )
                fold_maes.append(mae(y_val_raw, from_train_pred(m.predict(X_val))))
            return float(np.mean(fold_maes))

        optuna_start = time.time()
        # [Fix 2] TPESampler seed 고정
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )

        def optuna_cb(study, trial):
            print(
                f"  trial {trial.number + 1:>2}/{OPTUNA_TRIALS}"
                f"  MAE {trial.value:.4f}  best {study.best_value:.4f}"
                f"  {elapsed(optuna_start)}",
                flush=True,
            )

        study.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, callbacks=[optuna_cb])
        best_hp = study.best_params
        print(f"\n  최적 파라미터: {best_hp}")
        print(f"  Optuna 소요: {elapsed(optuna_start)}")
    else:
        if USE_OPTUNA_HPO and not OPTUNA_AVAILABLE:
            print("  Optuna가 설치되어 있지 않아 기본 LGB 파라미터를 사용합니다.")
        best_hp = dict(
            learning_rate=0.01, max_depth=-1, num_leaves=2047,
            min_child_samples=60, subsample=0.75,
            colsample_bytree=0.5, reg_alpha=0.3, reg_lambda=5.0,
        )

    # ── 7. Stage 1 Model Parameters ──
    lgb_params = dict(
        objective="regression_l1",
        n_estimators=25000,
        learning_rate=best_hp.get("learning_rate", 0.01),
        max_depth=best_hp.get("max_depth", -1),
        num_leaves=best_hp.get("num_leaves", 2047),
        min_child_samples=best_hp.get("min_child_samples", 60),
        subsample=best_hp.get("subsample", 0.75),
        subsample_freq=1,
        colsample_bytree=best_hp.get("colsample_bytree", 0.5),
        reg_alpha=best_hp.get("reg_alpha", 0.3),
        reg_lambda=best_hp.get("reg_lambda", 5.0),
        random_state=SEED, verbose=-1,
    )
    xgb_params = dict(
        objective="reg:absoluteerror",
        n_estimators=20000,
        learning_rate=min(best_hp.get("learning_rate", 0.015) * 1.2, 0.03),
        max_depth=min(best_hp.get("max_depth", 10), 10),
        subsample=best_hp.get("subsample", 0.75),
        colsample_bytree=best_hp.get("colsample_bytree", 0.5),
        colsample_bynode=0.5,
        reg_alpha=best_hp.get("reg_alpha", 0.3),
        reg_lambda=max(best_hp.get("reg_lambda", 3.0), 2.0),
        random_state=SEED, tree_method="hist",
        eval_metric="mae", early_stopping_rounds=500, verbosity=0,
    )
    cat_params = dict(
        iterations=20000,
        learning_rate=min(best_hp.get("learning_rate", 0.015) * 1.2, 0.03),
        depth=min(best_hp.get("max_depth", 10), 10),
        l2_leaf_reg=5.0, bootstrap_type="MVS",
        subsample=best_hp.get("subsample", 0.75),
        colsample_bylevel=best_hp.get("colsample_bytree", 0.5),
        loss_function="MAE", eval_metric="MAE",
        random_seed=SEED, task_type="CPU", early_stopping_rounds=500,
    )

    # ── 8. Stage 1 Training ──
    section(f"Stage 1 학습 (LGB x{len(LGB_SEEDS)} seeds + XGB + CAT)")
    t0 = time.time()
    oof_s1_lgb = np.zeros(len(train))
    oof_s1_xgb = np.zeros(len(train))
    oof_s1_cat = np.zeros(len(train))
    models_s1_lgb, models_s1_xgb, models_s1_cat = [], [], []
    ae_fold_artifacts = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y_strat, groups=groups), 1):
        print(f"\n  ── Fold {fold}/{N_FOLDS} ──")
        tr_df = train.iloc[tr_idx].copy()
        va_df = train.iloc[va_idx].copy()

        if USE_AUTOENCODER:
            med      = tr_df[ae_input_cols].median()
            X_ae_tr  = ae_prepare_matrix(tr_df, ae_input_cols, med)
            X_ae_va  = ae_prepare_matrix(va_df, ae_input_cols, med)
            sc, sd, z_va = ae_train_fold(X_ae_tr, X_ae_va, AE_DEVICE, SEED + fold)
            train.loc[train.index[va_idx], AE_COLS] = z_va
            va_df[AE_COLS] = z_va
            ae_fold_artifacts.append((sc, sd))
            print(f"    AE: val embedding {z_va.shape}")

        X_tr  = tr_df[feature_cols_s1]
        X_va  = va_df[feature_cols_s1]
        y_tr  = y_all[tr_idx]
        sw_tr = sw_all[tr_idx]

        lgb_preds = []
        for rs in LGB_SEEDS:
            lp = {**lgb_params, "random_state": rs}
            m  = lgb.LGBMRegressor(**lp)
            m.fit(
                X_tr, y_tr, sample_weight=sw_tr,
                eval_set=[(X_va, y_all[va_idx])], eval_metric="mae",
                callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)],
            )
            lgb_preds.append(from_train_pred(m.predict(X_va)))
            models_s1_lgb.append(m)
        oof_s1_lgb[va_idx] = np.mean(lgb_preds, axis=0)

        try:
            m_xgb = xgb.XGBRegressor(**xgb_params)
            m_xgb.fit(X_tr, y_tr, sample_weight=sw_tr,
                      eval_set=[(X_va, y_all[va_idx])], verbose=False)
        except Exception:
            xgb_fb = dict(xgb_params)
            xgb_fb["objective"] = "reg:squarederror"
            m_xgb = xgb.XGBRegressor(**xgb_fb)
            m_xgb.fit(X_tr, y_tr, sample_weight=sw_tr,
                      eval_set=[(X_va, y_all[va_idx])], verbose=False)
        oof_s1_xgb[va_idx] = from_train_pred(m_xgb.predict(X_va))
        models_s1_xgb.append(m_xgb)

        m_cat = cb.CatBoostRegressor(**cat_params)
        m_cat.fit(
            X_tr, y_tr, sample_weight=sw_tr,
            eval_set=(X_va, y_all[va_idx]),
            verbose=max(1, cat_params["iterations"] // 5), use_best_model=True,
        )
        oof_s1_cat[va_idx] = from_train_pred(m_cat.predict(X_va))
        models_s1_cat.append(m_cat)

        avg3 = (oof_s1_lgb[va_idx] + oof_s1_xgb[va_idx] + oof_s1_cat[va_idx]) / 3
        print(
            f"    Fold {fold} MAE: LGB={mae(y_raw[va_idx], oof_s1_lgb[va_idx]):.4f}"
            f"  XGB={mae(y_raw[va_idx], oof_s1_xgb[va_idx]):.4f}"
            f"  CAT={mae(y_raw[va_idx], oof_s1_cat[va_idx]):.4f}"
            f"  AVG={mae(y_raw[va_idx], avg3):.4f}"
        )

    mae_s1    = {"lgb": mae(y_raw, oof_s1_lgb), "xgb": mae(y_raw, oof_s1_xgb), "cat": mae(y_raw, oof_s1_cat)}
    oof_s1_by = {"lgb": oof_s1_lgb, "xgb": oof_s1_xgb, "cat": oof_s1_cat}
    print(f"\n  S1 OOF: LGB={mae_s1['lgb']:.4f}  XGB={mae_s1['xgb']:.4f}  CAT={mae_s1['cat']:.4f}")

    best_s1_models, best_s1_p, best_s1_mae = None, None, float("inf")
    for models in _powerset_models():
        for p in P_GRID:
            pred  = ensemble_pred(oof_s1_by, mae_s1, models, p)
            m_val = mae(y_raw, pred)
            if m_val < best_s1_mae:
                best_s1_mae, best_s1_models, best_s1_p = m_val, list(models), p

    w_s1  = {m: 1.0 / (mae_s1[m] ** best_s1_p) for m in best_s1_models}
    ws_s1 = sum(w_s1.values())
    oof_s1 = sum(w_s1[m] * oof_s1_by[m] for m in best_s1_models) / ws_s1
    print(f"  ★ S1 Ensemble: {best_s1_models} p={best_s1_p}  OOF MAE={best_s1_mae:.6f}  ({elapsed(t0)})")

    if USE_AUTOENCODER and ae_fold_artifacts:
        med_te   = train[ae_input_cols].median()
        X_te_raw = ae_prepare_matrix(test, ae_input_cols, med_te)
        z_folds  = [ae_encode(sc.transform(X_te_raw), sd, AE_DEVICE) for sc, sd in ae_fold_artifacts]
        z_mean   = np.mean(z_folds, axis=0)
        for i, c in enumerate(AE_COLS):
            test[c] = z_mean[:, i]

    # ── 9. Stage 2 ──
    if USE_STAGE2:
        section("Stage 2 — Pred-lag stack")
        t0    = time.time()
        train = add_pred_lag_features(train, oof_s1, global_mean)
        feature_cols_s2 = feature_cols_s1 + PRED_LAG_COLS
        print(f"  S2 피처: {len(feature_cols_s2)} (+pred_lag {len(PRED_LAG_COLS)})")

        lgb_params_s2 = dict(
            objective="regression_l1", n_estimators=12000, learning_rate=0.02,
            max_depth=-1, num_leaves=511, min_child_samples=80,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.65,
            reg_alpha=0.15, reg_lambda=4.0, random_state=SEED, verbose=-1,
        )
        xgb_params_s2 = dict(
            objective="reg:absoluteerror", n_estimators=8000, learning_rate=0.02,
            max_depth=6, subsample=0.8, colsample_bytree=0.65, colsample_bynode=0.5,
            reg_alpha=0.15, reg_lambda=2.5, random_state=SEED, tree_method="hist",
            eval_metric="mae", early_stopping_rounds=120, verbosity=0,
        )
        cat_params_s2 = dict(
            iterations=8000, learning_rate=0.02, depth=6, l2_leaf_reg=4.0,
            bootstrap_type="MVS", subsample=0.8, colsample_bylevel=0.65,
            loss_function="MAE", eval_metric="MAE",
            random_seed=SEED, task_type="CPU", early_stopping_rounds=120,
        )

        oof_s2_lgb = np.zeros(len(train))
        oof_s2_xgb = np.zeros(len(train))
        oof_s2_cat = np.zeros(len(train))
        models_s2_lgb, models_s2_xgb, models_s2_cat = [], [], []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y_strat, groups=groups), 1):
            print(f"\n  ── S2 Fold {fold}/{N_FOLDS} ──")
            tr_df = train.iloc[tr_idx]
            va_df = train.iloc[va_idx]
            X_tr  = tr_df[feature_cols_s2].replace([np.inf, -np.inf], np.nan)
            X_va  = va_df[feature_cols_s2].replace([np.inf, -np.inf], np.nan)
            assert_numeric_features(X_tr, X_tr.columns, "Stage2 X_tr")
            assert_numeric_features(X_va, X_va.columns, "Stage2 X_va")
            y_tr  = y_all[tr_idx]
            y_va  = y_all[va_idx]
            sw_tr = sw_all[tr_idx]

            m_lgb = lgb.LGBMRegressor(**lgb_params_s2)
            m_lgb.fit(
                X_tr, y_tr, sample_weight=sw_tr,
                eval_set=[(X_va, y_va)], eval_metric="mae",
                callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(-1)],
            )
            oof_s2_lgb[va_idx] = from_train_pred(m_lgb.predict(X_va))
            models_s2_lgb.append(m_lgb)

            try:
                m_xgb = xgb.XGBRegressor(**xgb_params_s2)
                m_xgb.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False)
            except Exception:
                xgb_fb = dict(xgb_params_s2)
                xgb_fb["objective"] = "reg:squarederror"
                m_xgb = xgb.XGBRegressor(**xgb_fb)
                m_xgb.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_s2_xgb[va_idx] = from_train_pred(m_xgb.predict(X_va))
            models_s2_xgb.append(m_xgb)

            m_cat = cb.CatBoostRegressor(**cat_params_s2)
            m_cat.fit(
                X_tr, y_tr, sample_weight=sw_tr,
                eval_set=(X_va, y_va), verbose=False, use_best_model=True,
            )
            oof_s2_cat[va_idx] = from_train_pred(m_cat.predict(X_va))
            models_s2_cat.append(m_cat)

            avg3 = (oof_s2_lgb[va_idx] + oof_s2_xgb[va_idx] + oof_s2_cat[va_idx]) / 3
            print(
                f"    LGB={mae(y_raw[va_idx], oof_s2_lgb[va_idx]):.4f}"
                f"  XGB={mae(y_raw[va_idx], oof_s2_xgb[va_idx]):.4f}"
                f"  CAT={mae(y_raw[va_idx], oof_s2_cat[va_idx]):.4f}"
                f"  AVG={mae(y_raw[va_idx], avg3):.4f}"
            )

        mae_s2    = {"lgb": mae(y_raw, oof_s2_lgb), "xgb": mae(y_raw, oof_s2_xgb), "cat": mae(y_raw, oof_s2_cat)}
        oof_s2_by = {"lgb": oof_s2_lgb, "xgb": oof_s2_xgb, "cat": oof_s2_cat}
        print(f"\n  S2 OOF: LGB={mae_s2['lgb']:.4f}  XGB={mae_s2['xgb']:.4f}  CAT={mae_s2['cat']:.4f}")

        best_s2_models, best_s2_p, best_s2_mae = None, None, float("inf")
        for models in _powerset_models():
            for p in P_GRID:
                pred  = ensemble_pred(oof_s2_by, mae_s2, models, p)
                m_val = mae(y_raw, pred)
                if m_val < best_s2_mae:
                    best_s2_mae, best_s2_models, best_s2_p = m_val, list(models), p

        w_s2  = {m: 1.0 / (mae_s2[m] ** best_s2_p) for m in best_s2_models}
        ws_s2 = sum(w_s2.values())
        oof_s2 = sum(w_s2[m] * oof_s2_by[m] for m in best_s2_models) / ws_s2
        print(f"  ★ S2 Ensemble: {best_s2_models} p={best_s2_p}  OOF MAE={best_s2_mae:.6f}  ({elapsed(t0)})")
        print(f"\n  S1 → S2 개선: {best_s1_mae - best_s2_mae:.6f}")

    # ── 10. Blending + Clipping ──
    section("Blending + Clipping")
    if USE_STAGE2:
        best_alpha, best_blend_mae = 0.0, mae(y_raw, oof_s2)
        for alpha in np.arange(0.0, 1.01, BLEND_ALPHA_STEP):
            blend = alpha * oof_s1 + (1.0 - alpha) * oof_s2
            m_val = mae(y_raw, blend)
            if m_val < best_blend_mae:
                best_blend_mae, best_alpha = m_val, float(alpha)
        oof_blend = best_alpha * oof_s1 + (1.0 - best_alpha) * oof_s2
    else:
        best_alpha     = 1.0
        oof_blend      = oof_s1.copy()
        best_blend_mae = best_s1_mae

    clip_hi   = float(np.percentile(y_raw, 100 * CLIP_Q))
    oof_final = np.clip(oof_blend, 0.0, clip_hi)
    final_mae = mae(y_raw, oof_final)

    # [Fix 11] timeslot alpha / global calibration 비활성화 (OOF 과적합 방지)
    # ENABLE_TIMESLOT_ALPHA_SEARCH = False, ENABLE_GLOBAL_CALIBRATION_SEARCH = False
    FINAL_METHOD    = "global_alpha"
    ts_alpha_map    = {}
    calib_params    = {"scale": 1.0, "bias": 0.0, "clip_q": None, "clip_upper": None}
    USE_CALIBRATION = False

    print(f"  blend alpha={best_alpha:.2f} (S1 weight)")
    print(f"  clip_q={CLIP_Q} (hi={clip_hi:.2f})")
    print(f"  [Fix 11] timeslot_alpha_search=OFF  global_calibration=OFF")
    print(f"\n  ▶ S1 OOF MAE:  {best_s1_mae:.6f}")
    if USE_STAGE2:
        print(f"  ▶ S2 OOF MAE:  {best_s2_mae:.6f}")
    print(f"  ▶ Blend MAE:   {best_blend_mae:.6f}")
    print(f"  ★★ FINAL OOF MAE: {final_mae:.6f}")

    # ── 11. Test Prediction ──
    section("Test 예측 + 제출")
    test = test.sort_values(["scenario_id", "ID"]).reset_index(drop=True)

    X_test_s1 = test[feature_cols_s1]
    p_lgb = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_lgb], axis=0)
    p_xgb = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_xgb], axis=0)
    p_cat = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_cat], axis=0)
    pred_s1 = sum(w_s1[m] * {"lgb": p_lgb, "xgb": p_xgb, "cat": p_cat}[m] for m in best_s1_models) / ws_s1
    print(f"  S1 test: mean={pred_s1.mean():.2f}  std={pred_s1.std():.2f}")

    if USE_STAGE2:
        test    = add_pred_lag_features(test, pred_s1, global_mean)
        X_test_s2 = test[feature_cols_s2]
        p2_lgb  = np.mean([from_train_pred(m.predict(X_test_s2)) for m in models_s2_lgb], axis=0)
        p2_xgb  = np.mean([from_train_pred(m.predict(X_test_s2)) for m in models_s2_xgb], axis=0)
        p2_cat  = np.mean([from_train_pred(m.predict(X_test_s2)) for m in models_s2_cat], axis=0)
        pred_s2 = sum(w_s2[m] * {"lgb": p2_lgb, "xgb": p2_xgb, "cat": p2_cat}[m] for m in best_s2_models) / ws_s2
        print(f"  S2 test: mean={pred_s2.mean():.2f}  std={pred_s2.std():.2f}")
        pred_blend = best_alpha * pred_s1 + (1.0 - best_alpha) * pred_s2
    else:
        pred_blend = pred_s1

    pred_final = np.clip(pred_blend, 0.0, clip_hi)

    # [Fix 5] 상대 경로 저장
    sub = pd.DataFrame({"ID": test["ID"], TARGET: pred_final})
    sub = sub.sort_values("ID").reset_index(drop=True)
    save_path = "submission_v17_noleak.csv"
    sub.to_csv(save_path, index=False)

    oof_path = "oof_v17_noleak.csv"
    pd.DataFrame({
        "ID":        train["ID"].values,
        "y_true":    y_raw,
        "oof_pred":  oof_final,
        "abs_error": np.abs(y_raw - oof_final),
    }).to_csv(oof_path, index=False)

    print(f"\n  ▶ 저장: {save_path}")
    print(f"  ▶ OOF 저장: {oof_path}")
    print(f"  ▶ 전체 소요: {elapsed(T0)}")
    print(f"\n  ★★ FINAL OOF MAE: {final_mae:.6f}")


if __name__ == "__main__":
    main()
