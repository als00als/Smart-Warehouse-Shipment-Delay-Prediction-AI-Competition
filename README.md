# 스마트 창고 출고 지연 예측 AI 경진대회 솔루션

**TEAM jiwonyyy**

---

## 결과

| | 점수 |
|---|---|
| OOF MAE | 8.458114 |
| Public Score | 9.71506 |
| Private Score | 9.9564 |

---

## 실행 방법

스크립트와 같은 폴더 또는 `data/` 서브폴더에 아래 파일을 위치시킨 후 실행합니다.

```
train.csv
test.csv
layout_info.csv
```

```bash
python SmartWareHouse_v18.py
```

출력 파일: `submission_v17_hybrid_best_combo.csv`

---

## 개발 환경

| 항목 | 버전 |
|---|---|
| OS | Windows 11 Home |
| Python | 3.14.0 |
| LightGBM | 4.3.0 |
| XGBoost | 2.0.3 |
| CatBoost | 1.2.5 |
| PyTorch | 2.2.2 |
| scikit-learn | 1.4.2 |
| Optuna | 3.6.1 |
| NumPy | 1.26.4 |
| Pandas | 2.2.2 |

---

## 전체 파이프라인

```
STEP 01  데이터 로드 & 결측치 처리
STEP 02  피처 엔지니어링
STEP 03  AutoEncoder — 잠재 피처 추출
STEP 04  Target Encoding
STEP 05  Stage 1 학습 (LGB×2 seeds + XGB + CAT)
STEP 06  Stage 2 학습 (Pred-lag 피처 추가 후 재학습)
STEP 07  Power-weighted 앙상블 + Global Calibration
STEP 08  결과 추출
```

---

## 피처 엔지니어링

### 기본 파생 피처
- `robot_efficiency` = robot_active / robot_total
- `risk_index` = congestion_score × (1 − robot_efficiency)
- `bottle_neck` = order_per_station × congestion_score
- `battery_risk`, `trip_per_robot`, `order_pressure` 등

### 시계열 피처 (scenario_id 기준)
- **Lag**: t-1, t-2, t-3
- **Diff**: 1차, 2차 차분
- **Rolling**: window 3, 5, 10 (mean / std)
- **EWM**: alpha=0.3 지수 가중 이동 평균
- 적용 대상: congestion_score, order_inflow_15m 등 핵심 13개 컬럼

### 시나리오 집계 피처
- 시나리오 단위 mean / max / min / std
- 현재값 / 시나리오 평균 (`rel_to_scen`)
- 시나리오 내 분위수 rank

### Target Encoding
- 대상: `layout_id`, `timeslot`, `layout_type` 및 교차 조합
- GroupKFold OOF 방식으로 타겟 누수 방지
- Smoothing = 20

### AutoEncoder 임베딩
- 구조: Input(~300) → Hidden(256) → Latent(32) → Hidden(256) → Output
- 폴드별 훈련 데이터로만 학습 → 검증 폴드 인코딩 (Data Leak 방지)
- 32차원 `ae_z0 ~ ae_z31` 피처 추가
- Early Stopping (patience=5) + StandardScaler 정규화

---

## 모델 구성

### Optuna HPO
- LightGBM 하이퍼파라미터 자동 탐색
- 20 trials, TPESampler(seed=42)
- 전체 학습 데이터의 40% 샘플로 빠른 탐색
- 탐색 파라미터: learning_rate, num_leaves, min_child_samples, reg_lambda 등 8개

### Stage 1

| 파라미터 | LightGBM | XGBoost | CatBoost |
|---|---|---|---|
| n_estimators | 25,000 | 20,000 | 20,000 |
| learning_rate | Optuna | Optuna × 1.2 | Optuna × 1.2 |
| num_leaves | Optuna | — | — |
| depth / max_depth | Optuna | Optuna (max 10) | Optuna (max 10) |
| subsample | Optuna | Optuna | Optuna |
| reg_lambda | Optuna | Optuna | 5.0 |
| early_stopping | 300 rounds | 500 rounds | 500 rounds |
| seeds | 42, 142 | 42 | 42 |

### Stage 2 (Pred-lag Stacking)

Stage 1 OOF 예측값으로 시계열 피처 9개 생성 후 재학습

```
pred_lag1/2/3, pred_diff1/2,
pred_roll3_mean, pred_roll5_mean, pred_ewm3, pred_lag1_log
```

| 파라미터 | LightGBM | XGBoost | CatBoost |
|---|---|---|---|
| n_estimators | 12,000 | 8,000 | 8,000 |
| learning_rate | 0.02 | 0.02 | 0.02 |
| depth / max_depth | — | 6 | 6 |
| num_leaves | 511 | — | — |
| subsample | 0.8 | 0.8 | 0.8 |
| reg_lambda | 4.0 | 2.5 | 4.0 |
| early_stopping | 120 rounds | 120 rounds | 120 rounds |

---

## 앙상블 & 후처리

### Power-weighted Ensemble
- `weight(m) = 1 / MAE(m)^p`
- p ∈ {1.0, 1.5, 2.0, 3.0, 4.0} 자동 탐색
- LGB / XGB / CAT 7가지 조합 전수 탐색 → OOF MAE 최저 선택

### S1 ↔ S2 Blending
- `최종 = α × S1 + (1-α) × S2`
- α를 0~1 범위에서 자동 탐색 (step=0.01)
- 타임슬롯별 최적 α 개별 적용 → 20% 글로벌 스무딩

### Global Calibration
- scale (0.94~1.06), bias (-1.0~1.0), clip_q 조합 탐색
- OOF MAE 개선될 때만 적용 → Public LB 직접 튜닝 금지

---

## 검증 전략

- `StratifiedGroupKFold` (5-fold, shuffle=True, seed=42)
- Group 기준: `scenario_id` → 시나리오 단위로 완전 분리
- Stratify 기준: 시나리오 평균 타겟 10분위 → 폴드별 지연 분포 균등

---

## 재현성 & Seed 고정

모든 난수 소스에 seed=42 고정

| 구성요소 | 설정 |
|---|---|
| StratifiedGroupKFold | random_state=42 |
| Optuna TPESampler | seed=42 |
| Optuna 샘플 선택 | np.default_rng(42) |
| LightGBM | random_state=42 |
| XGBoost | random_state=42 |
| CatBoost | random_seed=42 |
| AutoEncoder (PyTorch) | manual_seed(42+fold) |

> 리더보드 제출 시점에는 Optuna seed 미고정 상태였으며,
> 코드 제출 규칙("Private Score 복원 가능") 준수를 위해 seed를 고정했습니다.

---

## 한계점 & 개선 방향

### 한계점
- OOF-LB 갭 약 1.27 — 시나리오 집계 피처의 훈련 시나리오 과적합
- AutoEncoder가 훈련 시나리오 분포 학습 → 미관측 시나리오에서 임베딩 품질 저하 가능
- Calibration 탐색 공간(1,350 조합)이 넓어 OOF 과적합 가능성

### 향후 개선 방향
- 시나리오 집계에서 max/min 제거, mean/std만 유지
- AutoEncoder dropout 추가 (0.0 → 0.2), latent dim 축소
- N_Folds 5 → 7로 증가
- Layout 기준 GroupKFold로 분포 시프트 직접 대응
