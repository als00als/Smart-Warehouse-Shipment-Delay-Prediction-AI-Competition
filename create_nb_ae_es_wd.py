"""
Duplicate:
  스마트 창고 출고 지연 예측_0416_preproc_logtarget_s2_predlag_autoencoder.ipynb
into:
  스마트 창고 출고 지연 예측_0416_preproc_logtarget_s2_predlag_autoencoder_ae_es_wd.ipynb

Changes:
- AE: validation MSE early stopping (best state restore), max epochs = AE_EPOCHS
- Adam weight_decay = AE_WEIGHT_DECAY
- Markdown note + distinct submission filename
"""
import copy
import glob
import json
import re
from pathlib import Path


def _to_source_lines(text: str):
    lines = text.split("\n")
    if not lines:
        return []
    out = [f"{line}\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


OLD_TRAIN = """def _ae_train_fold(X_tr: np.ndarray, X_va: np.ndarray, device: torch.device, seed: int):
    torch.manual_seed(seed)
    n_in = X_tr.shape[1]
    model = TabularAutoEncoder(n_in, AE_HIDDEN_DIM, AE_LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    tr_t = torch.from_numpy(X_tr_s).float().to(device)
    va_t = torch.from_numpy(X_va_s).float().to(device)
    n = tr_t.shape[0]
    for _ in range(AE_EPOCHS):
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
    model.eval()
    with torch.no_grad():
        _, z_va = model(va_t)
    z_va_np = z_va.cpu().numpy()
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return scaler, state_cpu, z_va_np"""


NEW_TRAIN = """def _ae_train_fold(X_tr: np.ndarray, X_va: np.ndarray, device: torch.device, seed: int):
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
    return scaler, state_cpu, z_va_np"""


def main():
    src_matches = glob.glob("*0416_preproc_logtarget_s2_predlag_autoencoder.ipynb")
    if not src_matches:
        raise FileNotFoundError("Source notebook not found")
    src = Path(src_matches[0])
    out = src.with_name(
        "스마트 창고 출고 지연 예측_0416_preproc_logtarget_s2_predlag_autoencoder_ae_es_wd.ipynb"
    )

    nb = json.loads(src.read_text(encoding="utf-8"))
    new_nb = copy.deepcopy(nb)

    # --- markdown intro (cell 1) ---
    md = "".join(new_nb["cells"][1].get("source", []))
    md = md.replace(
        "### 스마트 창고 출고 지연 예측 — AutoEncoder + Stage1/2 (0416)\n",
        "### 스마트 창고 출고 지연 예측 — AutoEncoder + Stage1/2 (0416, AE ES+WD)\n",
        1,
    )
    if "- **이 변형(variant)**:" not in md:
        md = md.replace(
            "- 끄기: `USE_AUTOENCODER = False`\n",
            "- 끄기: `USE_AUTOENCODER = False`\n"
            "- **이 변형(variant)**: fold 검증 MSE **early stopping**(`AE_ES_PATIENCE`, `AE_ES_MIN_DELTA`) + "
            "최저 val 시점 **best state 복원**. Adam `weight_decay=AE_WEIGHT_DECAY`.\n"
            "- `AE_EPOCHS`: early stopping **최대 에폭(상한)**.\n",
            1,
        )
    new_nb["cells"][1]["source"] = _to_source_lines(md)

    # --- config cell (index 3) ---
    c3 = "".join(new_nb["cells"][3].get("source", []))
    needle = 'AE_LR = 1e-3\n# CUDA 사용 가능 시 GPU (대용량 행렬에서 유리)\nAE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    insert = (
        "AE_LR = 1e-3\n"
        "# AE_EPOCHS: early stopping 최대 에폭(상한). val MSE가 patience 동안 개선 없으면 조기 종료.\n"
        "AE_ES_PATIENCE = 4\n"
        "AE_ES_MIN_DELTA = 0.0\n"
        "AE_WEIGHT_DECAY = 1e-4\n"
        '# CUDA 사용 가능 시 GPU (대용량 행렬에서 유리)\n'
        'AE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    )
    if needle not in c3:
        raise RuntimeError("Config cell AE_LR/AE_DEVICE block not found; notebook layout changed.")
    c3 = c3.replace(needle, insert, 1)
    new_nb["cells"][3]["source"] = _to_source_lines(c3)

    # --- helpers cell (index 5): replace _ae_train_fold ---
    c5 = "".join(new_nb["cells"][5].get("source", []))
    if OLD_TRAIN not in c5:
        raise RuntimeError("_ae_train_fold block not found; notebook may already be patched.")
    c5 = c5.replace(OLD_TRAIN, NEW_TRAIN, 1)
    new_nb["cells"][5]["source"] = _to_source_lines(c5)

    # --- submission filename (last code cell) ---
    last_idx = None
    for i in range(len(new_nb["cells"]) - 1, -1, -1):
        if new_nb["cells"][i].get("cell_type") == "code" and "save_path = os.path.join" in "".join(
            new_nb["cells"][i].get("source", [])
        ):
            last_idx = i
            break
    if last_idx is None:
        raise RuntimeError("Could not find save_path cell")
    last_src = "".join(new_nb["cells"][last_idx].get("source", []))
    last_src, n = re.subn(
        r"save_path = os\.path\.join\(project_root, '[^']+'\)",
        "save_path = os.path.join(project_root, 'submission_logtarget_s2_predlag_autoencoder_ae_es_wd.csv')",
        last_src,
        count=1,
    )
    if n != 1:
        raise RuntimeError(f"save_path replace failed (n={n})")
    new_nb["cells"][last_idx]["source"] = _to_source_lines(last_src)

    out.write_text(json.dumps(new_nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
