"""
Apply MAE-oriented upgrades to:
  스마트 창고 출고 지연 예측_0416_preproc_logtarget_s2_predlag_autoencoder_ae_es_wd.ipynb

In-place edits:
- AE: AdamW + optional denoise training + dropout; tuned ES/WD defaults; higher epoch cap
- Stage1b residual: gated by USE_STAGE1_RESIDUAL (default ON); test cell respects flag
- BLEND_ALPHA_STEP 0.005; slightly denser CLIP_Q_GRID
- Markdown intro bullets
"""
import json
import glob
from pathlib import Path


def _to_source_lines(text: str):
    lines = text.split("\n")
    if not lines:
        return []
    out = [f"{line}\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


OLD_CLASS_AND_TRAIN = '''class TabularAutoEncoder(nn.Module):
    def __init__(self, n_in: int, hidden: int, latent: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_in),
        )

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat, z


def _ae_prepare_matrix(df: pd.DataFrame, cols: list[str], medians: pd.Series) -> np.ndarray:
    X = df[cols].to_numpy(dtype=np.float64)
    for j, col in enumerate(cols):
        m = float(medians[col])
        colv = X[:, j]
        colv[~np.isfinite(colv)] = m
        colv[np.isnan(colv)] = m
    return X


def _ae_train_fold(X_tr: np.ndarray, X_va: np.ndarray, device: torch.device, seed: int):
    torch.manual_seed(seed)
    n_in = X_tr.shape[1]
    model = TabularAutoEncoder(n_in, AE_HIDDEN_DIM, AE_LATENT_DIM).to(device)
    wd = float(globals().get("AE_WEIGHT_DECAY", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=wd)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    tr_t = torch.from_numpy(X_tr_s).float().to(device)
    va_t = torch.from_numpy(X_va_s).float().to(device)
    n = tr_t.shape[0]

    max_epochs = int(AE_EPOCHS)
    patience = int(globals().get("AE_ES_PATIENCE", 4))
    min_delta = float(globals().get("AE_ES_MIN_DELTA", 0.0))

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for _epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, AE_BATCH_SIZE):
            idx = perm[i : i + AE_BATCH_SIZE]
            xb = tr_t[idx]
            opt.zero_grad(set_to_none=True)
            xh, _ = model(xb)
            loss = nn.functional.mse_loss(xh, xb)
            loss.backward()
            opt.step()

        if va_t.shape[0] == 0:
            break

        model.eval()
        with torch.no_grad():
            xh_va, z_va = model(va_t)
            val_loss = nn.functional.mse_loss(xh_va, va_t, reduction="mean").item()

        if val_loss < (best_val - min_delta):
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        _, z_va = model(va_t)
    z_va_np = z_va.cpu().numpy()
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return scaler, state_cpu, z_va_np'''


NEW_CLASS_AND_TRAIN = '''class TabularAutoEncoder(nn.Module):
    def __init__(self, n_in: int, hidden: int, latent: int, dropout: float = 0.0):
        super().__init__()
        p = float(dropout)
        enc_layers = [nn.Linear(n_in, hidden), nn.ReLU(inplace=True)]
        if p > 0:
            enc_layers.append(nn.Dropout(p))
        enc_layers.append(nn.Linear(hidden, latent))
        self.enc = nn.Sequential(*enc_layers)
        dec_layers = [nn.Linear(latent, hidden), nn.ReLU(inplace=True)]
        if p > 0:
            dec_layers.append(nn.Dropout(p))
        dec_layers.append(nn.Linear(hidden, n_in))
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat, z


def _ae_prepare_matrix(df: pd.DataFrame, cols: list[str], medians: pd.Series) -> np.ndarray:
    X = df[cols].to_numpy(dtype=np.float64)
    for j, col in enumerate(cols):
        m = float(medians[col])
        colv = X[:, j]
        colv[~np.isfinite(colv)] = m
        colv[np.isnan(colv)] = m
    return X


def _ae_train_fold(X_tr: np.ndarray, X_va: np.ndarray, device: torch.device, seed: int):
    torch.manual_seed(seed)
    n_in = X_tr.shape[1]
    p_drop = float(globals().get("AE_DROPOUT", 0.0))
    model = TabularAutoEncoder(n_in, AE_HIDDEN_DIM, AE_LATENT_DIM, p_drop).to(device)
    wd = float(globals().get("AE_WEIGHT_DECAY", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=AE_LR, weight_decay=wd)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    tr_t = torch.from_numpy(X_tr_s).float().to(device)
    va_t = torch.from_numpy(X_va_s).float().to(device)
    n = tr_t.shape[0]

    max_epochs = int(AE_EPOCHS)
    patience = int(globals().get("AE_ES_PATIENCE", 4))
    min_delta = float(globals().get("AE_ES_MIN_DELTA", 0.0))
    sigma = float(globals().get("AE_DENOISE_STD", 0.0))

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for _epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, AE_BATCH_SIZE):
            idx = perm[i : i + AE_BATCH_SIZE]
            xb = tr_t[idx]
            opt.zero_grad(set_to_none=True)
            if sigma > 0:
                x_in = xb + torch.randn_like(xb) * sigma
            else:
                x_in = xb
            xh, _ = model(x_in)
            loss = nn.functional.mse_loss(xh, xb)
            loss.backward()
            opt.step()

        if va_t.shape[0] == 0:
            break

        model.eval()
        with torch.no_grad():
            xh_va, z_va = model(va_t)
            val_loss = nn.functional.mse_loss(xh_va, va_t, reduction="mean").item()

        if val_loss < (best_val - min_delta):
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        _, z_va = model(va_t)
    z_va_np = z_va.cpu().numpy()
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return scaler, state_cpu, z_va_np'''


OLD_ENCODE_HEAD = '''def _ae_encode_matrix(X_scaled: np.ndarray, state_dict: dict, device: torch.device) -> np.ndarray:
    n_in = X_scaled.shape[1]
    model = TabularAutoEncoder(n_in, AE_HIDDEN_DIM, AE_LATENT_DIM).to(device)'''

NEW_ENCODE_HEAD = '''def _ae_encode_matrix(X_scaled: np.ndarray, state_dict: dict, device: torch.device) -> np.ndarray:
    n_in = X_scaled.shape[1]
    p_drop = float(globals().get("AE_DROPOUT", 0.0))
    model = TabularAutoEncoder(n_in, AE_HIDDEN_DIM, AE_LATENT_DIM, p_drop).to(device)'''


OLD_RESID_BLOCK = '''# Stage 1b: residual (OOF-safe) — base 예측 오차를 추가로 학습
section('Stage 1b - Residual LGB (stack on Stage1 ensemble)')
resid_params = dict(
    objective='regression_l1',
    n_estimators=8000,
    learning_rate=0.02,
    num_leaves=127,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=SEED,
    verbose=-1,
)
oof_s1_resid = np.zeros(len(train))
models_s1_resid = []
t1b = time.time()
for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):
    X_tr = train.iloc[tr_idx][feature_cols_s1]
    X_va = train.iloc[va_idx][feature_cols_s1]
    r_tr = y_raw[tr_idx] - oof_s1_pre[tr_idx]
    r_va = y_raw[va_idx] - oof_s1_pre[va_idx]
    sw_tr = sw_all[tr_idx]
    mr = lgb.LGBMRegressor(**resid_params)
    mr.fit(
        X_tr, r_tr,
        sample_weight=sw_tr,
        eval_set=[(X_va, r_va)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_s1_resid[va_idx] = mr.predict(X_va)
    models_s1_resid.append(mr)
    print(f"  S1b Fold {fold} residual MAE: {mae(r_va, oof_s1_resid[va_idx]):.6f}")

oof_s1 = oof_s1_pre + oof_s1_resid
s1_mae = mae(y_raw, oof_s1)
print(f"\\n▶ Stage 1 after residual OOF MAE: {s1_mae:.6f}  ({elapsed(t1b)})")'''


NEW_RESID_BLOCK = '''# Stage 1b: residual (OOF-safe) — base 예측 오차를 추가로 학습 (플래그로 ON/OFF)
if USE_STAGE1_RESIDUAL:
    section('Stage 1b - Residual LGB (stack on Stage1 ensemble)')
    resid_params = dict(
        objective='regression_l1',
        n_estimators=8000,
        learning_rate=0.02,
        num_leaves=127,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=SEED,
        verbose=-1,
    )
    oof_s1_resid = np.zeros(len(train))
    models_s1_resid = []
    t1b = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):
        X_tr = train.iloc[tr_idx][feature_cols_s1]
        X_va = train.iloc[va_idx][feature_cols_s1]
        r_tr = y_raw[tr_idx] - oof_s1_pre[tr_idx]
        r_va = y_raw[va_idx] - oof_s1_pre[va_idx]
        sw_tr = sw_all[tr_idx]
        mr = lgb.LGBMRegressor(**resid_params)
        mr.fit(
            X_tr, r_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, r_va)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_s1_resid[va_idx] = mr.predict(X_va)
        models_s1_resid.append(mr)
        print(f"  S1b Fold {fold} residual MAE: {mae(r_va, oof_s1_resid[va_idx]):.6f}")

    oof_s1 = oof_s1_pre + oof_s1_resid
    s1_mae = mae(y_raw, oof_s1)
    print(f"\\n▶ Stage 1 after residual OOF MAE: {s1_mae:.6f}  ({elapsed(t1b)})")
else:
    oof_s1_resid = np.zeros(len(train))
    models_s1_resid = []
    oof_s1 = np.asarray(oof_s1_pre, dtype=np.float64).copy()
    s1_mae = float(s1_pre_mae)
    print("\\n▶ Stage 1 residual: OFF  (oof_s1 = pre-ensemble only)")'''


OLD_TEST_S1 = '''pred_s1_pre = sum(w_s1[m] * {'lgb': p_lgb, 'xgb': p_xgb, 'cat': p_cat}[m] for m in best_s1_models) / ws_s1
pred_resid_test = np.mean([m.predict(X_test_s1) for m in models_s1_resid], axis=0)
pred_s1_test = pred_s1_pre + pred_resid_test
print(f"▶ Stage 1 test predictions ready (ensemble + residual)")'''

NEW_TEST_S1 = '''pred_s1_pre = sum(w_s1[m] * {'lgb': p_lgb, 'xgb': p_xgb, 'cat': p_cat}[m] for m in best_s1_models) / ws_s1
if USE_STAGE1_RESIDUAL and models_s1_resid:
    pred_resid_test = np.mean([m.predict(X_test_s1) for m in models_s1_resid], axis=0)
    pred_s1_test = pred_s1_pre + pred_resid_test
    print(f"▶ Stage 1 test predictions ready (ensemble + residual)")
else:
    pred_s1_test = pred_s1_pre
    print(f"▶ Stage 1 test predictions ready (ensemble only, residual OFF)")'''


OLD_FINAL_TEST_S1 = '''pred_s1_pre = sum(w_s1[m] * {'lgb': p_lgb_s1, 'xgb': p_xgb_s1, 'cat': p_cat_s1}[m]
                  for m in best_s1_models) / ws_s1
pred_resid_test = np.mean([m.predict(X_test_s1) for m in models_s1_resid], axis=0)
pred_s1_test = pred_s1_pre + pred_resid_test
print(f"  Stage 1 test pred: mean={pred_s1_test.mean():.2f}  std={pred_s1_test.std():.2f}")'''

NEW_FINAL_TEST_S1 = '''pred_s1_pre = sum(w_s1[m] * {'lgb': p_lgb_s1, 'xgb': p_xgb_s1, 'cat': p_cat_s1}[m]
                  for m in best_s1_models) / ws_s1
if USE_STAGE1_RESIDUAL and models_s1_resid:
    pred_resid_test = np.mean([m.predict(X_test_s1) for m in models_s1_resid], axis=0)
    pred_s1_test = pred_s1_pre + pred_resid_test
else:
    pred_s1_test = pred_s1_pre
print(f"  Stage 1 test pred: mean={pred_s1_test.mean():.2f}  std={pred_s1_test.std():.2f}")'''


def main():
    matches = glob.glob("*0416_preproc_logtarget_s2_predlag_autoencoder_ae_es_wd.ipynb")
    if not matches:
        raise FileNotFoundError("Notebook not found")
    path = Path(matches[0])
    nb = json.loads(path.read_text(encoding="utf-8"))

    # --- markdown cell 1 ---
    md = "".join(nb["cells"][1].get("source", []))
    extra = (
        "\n"
        "- **MAE boost 패키지**: `AdamW` + `AE_DENOISE_STD` + `AE_DROPOUT`, ES/WD 기본값 재튜닝, "
        "`BLEND_ALPHA_STEP=0.005`, 클립 q 그리드 보강, `USE_STAGE1_RESIDUAL`로 S1b 잔차 ON/OFF.\n"
    )
    if "MAE boost 패키지" not in md:
        md = md + extra
    nb["cells"][1]["source"] = _to_source_lines(md)

    # --- config cell 3 ---
    c3 = "".join(nb["cells"][3].get("source", []))
    c3 = c3.replace(
        "BLEND_ALPHA_STEP = 0.01  # fine alpha sweep\n",
        "BLEND_ALPHA_STEP = 0.005  # finer alpha sweep (S1–S2 blend)\n",
        1,
    )
    c3 = c3.replace(
        "# Stage1 과적합 완화: residual은 기본 OFF (필요 시 True)\nUSE_STAGE1_RESIDUAL = False\n",
        "# Stage1b residual (S1 앙상블 위 LGB 잔차). OFF면 학습/테스트 모두 pre-ensemble만 사용.\n"
        "USE_STAGE1_RESIDUAL = True\n",
        1,
    )
    c3 = c3.replace(
        "CLIP_Q_GRID = [0.992, 0.994, 0.995, 0.996, 0.998]\n",
        "CLIP_Q_GRID = [0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998]\n",
        1,
    )
    old_ae = (
        "AE_LATENT_DIM = 32\n"
        "AE_HIDDEN_DIM = 256\n"
        "AE_EPOCHS = 16\n"
        "AE_BATCH_SIZE = 4096\n"
        "AE_LR = 1e-3\n"
        "# AE_EPOCHS: early stopping 최대 에폭(상한). val MSE가 patience 동안 개선 없으면 조기 종료.\n"
        "AE_ES_PATIENCE = 4\n"
        "AE_ES_MIN_DELTA = 0.0\n"
        "AE_WEIGHT_DECAY = 1e-4\n"
    )
    new_ae = (
        "AE_LATENT_DIM = 32\n"
        "AE_HIDDEN_DIM = 256\n"
        "AE_BATCH_SIZE = 4096\n"
        "AE_LR = 1e-3\n"
        "# AE_EPOCHS: ES 최대 에폭(상한). val 재구성 MSE + patience + min_delta\n"
        "AE_EPOCHS = 48\n"
        "AE_ES_PATIENCE = 6\n"
        "AE_ES_MIN_DELTA = 1e-6\n"
        "AE_WEIGHT_DECAY = 3e-5\n"
        "AE_DENOISE_STD = 0.01\n"
        "AE_DROPOUT = 0.05\n"
    )
    if old_ae not in c3:
        raise RuntimeError("AE config block not found (expected 16-epoch defaults).")
    c3 = c3.replace(old_ae, new_ae, 1)
    nb["cells"][3]["source"] = _to_source_lines(c3)

    # --- helpers cell 5 ---
    c5 = "".join(nb["cells"][5].get("source", []))
    if OLD_CLASS_AND_TRAIN not in c5:
        raise RuntimeError("TabularAutoEncoder / _ae_train_fold block not found.")
    c5 = c5.replace(OLD_CLASS_AND_TRAIN, NEW_CLASS_AND_TRAIN, 1)
    if OLD_ENCODE_HEAD not in c5:
        raise RuntimeError("_ae_encode_matrix header not found.")
    c5 = c5.replace(OLD_ENCODE_HEAD, NEW_ENCODE_HEAD, 1)
    nb["cells"][5]["source"] = _to_source_lines(c5)

    # --- training cell 24 ---
    c24 = "".join(nb["cells"][24].get("source", []))
    if OLD_RESID_BLOCK not in c24:
        raise RuntimeError("Residual block not found in cell 24.")
    c24 = c24.replace(OLD_RESID_BLOCK, NEW_RESID_BLOCK, 1)
    if OLD_TEST_S1 not in c24:
        raise RuntimeError("pred_s1_test block not found in cell 24.")
    c24 = c24.replace(OLD_TEST_S1, NEW_TEST_S1, 1)
    nb["cells"][24]["source"] = _to_source_lines(c24)

    # --- final predict cell 39 ---
    c39 = "".join(nb["cells"][39].get("source", []))
    if OLD_FINAL_TEST_S1 not in c39:
        raise RuntimeError("Final test pred_s1 block not found in cell 39.")
    c39 = c39.replace(OLD_FINAL_TEST_S1, NEW_FINAL_TEST_S1, 1)
    nb["cells"][39]["source"] = _to_source_lines(c39)

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
