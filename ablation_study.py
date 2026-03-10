# ablation_study.py
# Phase 1 / Phase 2 / Phase 3 ablation 전용 스크립트
# 피험자 한 명(loso_sid)을 대상으로 3가지 조건을 순서대로 실행하고
# 결과를 DataFrame으로 비교 출력합니다.
#
# 실행 방법 (Kaggle / Colab 노트북):
#   exec(open("/kaggle/input/datasets/hdwmmmlc/codeupload/ablation_study.py").read())
#   run_ablation()
#
# 또는 터미널:
#   python ablation_study.py

import os
import sys
import copy
import types
import numpy as np

# ── 모듈 경로 설정 (train_stage2_manifold와 동일 로직) ───────────────────────
IS_KAGGLE: bool = (
    "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    or os.path.isdir("/kaggle/input")
)

def _setup_module_path() -> None:
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _here = None

    candidates = []
    if _here:
        candidates.append(_here)
    candidates.append(os.getcwd())
    if IS_KAGGLE:
        candidates.append("/kaggle/input/datasets/hdwmmmlc/codeupload")
        candidates.append("/kaggle/working")

    for _p in candidates:
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)

_setup_module_path()

import pandas as pd
import torch
from torch.utils.data import DataLoader
from train_stage2_manifold import (
    _build_subject_remap,
    train_phase1,
    train_phase2,
    train_phase3,
    extract_merged_features,
    extract_latent_features,
    DTCWTConfig,
    IS_KAGGLE as _IS_KAGGLE,
    _KAGGLE_DATA_DIR,
    _KAGGLE_CKPT_DIR,
)
from model_manifold import SpatialCNN, TemporalLSTM
from kara_one_dataset import loso_splits

TASK_NAMES = ["vowel", "nasal", "bilabial", "iy", "uw"]


def extract_raw_merged_features(n_channels: int, loader, device):
    """
    P1 encoder(fc layers)를 거치지 않고 CNN+LSTM의 merged feature만 추출.
    랜덤 초기화된 CNN+LSTM 사용 → No P1 ablation 조건의 베이스라인.
    출력 dim: 128 (SpatialCNN) + 1024 (TemporalLSTM) = 1152
    """
    cnn  = SpatialCNN(n_channels=n_channels).to(device).eval()
    lstm = TemporalLSTM(in_dim=n_channels).to(device).eval()

    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            eeg       = batch["eeg"].to(device)
            corr      = batch["corr"].to(device)
            corr_mask = batch.get("corr_mask", torch.ones_like(batch["corr"])).to(device)
            attn      = batch["attention_mask"].to(device)
            cnn_feat  = cnn(corr, corr_mask)        # [B, 128]
            lstm_feat = lstm(eeg, attn)              # [B, 1024]
            merged    = torch.cat([cnn_feat, lstm_feat], dim=1)  # [B, 1152]
            feats.append(merged.cpu().numpy())
            labels.append(batch["phon"].cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels)

def evaluate_phase1_only(model, loader, device):
    """P1 classifier head 출력으로 바로 accuracy 계산 (P2, P3 없이)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits, _, _, _, _ = model(
                batch["eeg"].to(device),
                batch["corr"].to(device),
                batch.get("corr_mask", torch.ones_like(batch["corr"])).to(device),
                batch["attention_mask"].to(device),
            )
            preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            all_preds.append(preds)
            all_labels.append(batch["phon"].numpy().astype(int))
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc_per_task = (preds == labels).mean(axis=0)
    return float(acc_per_task.mean()), acc_per_task


# ── 기본 설정 ─────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    # 데이터
    raw_data_dir  = _KAGGLE_DATA_DIR if IS_KAGGLE else "/data/kara_one",
    ckpt_dir      = (_KAGGLE_CKPT_DIR if IS_KAGGLE else "./checkpoints_ablation") + "/ablation",
    loso_sid      = 0,           # 피험자 한 명만
    seed          = 42,
    split_mode    = "loso_cv",
    channel_mode  = "paper10",
    n_channels    = 10,
    feature_mode  = "raw",
    ccv_repr      = "cov_bounded",
    ccv_reject_alpha   = 0.10,
    cov_bounded_scale  = 0.10,

    # ICA
    apply_ica          = False,
    ica_method         = "fastica",
    ica_n_components   = None,
    ica_random_state   = 97,
    ica_max_iter       = 512,

    # adversarial
    adv_weight         = 0.0,
    grl_lambda         = 0.5,
    adv_warmup_epochs  = 10,
    adv_val_weight     = 0.0,

    # Phase 1
    epochs_phase1      = 80,
    lr_phase1          = 1e-3,
    wd                 = 0.0,
    patience_phase1    = 8,
    latent_dim         = 32,
    batch_size         = 16,
    augment            = False,
    augment_start      = 0,
    use_task_pos_weight= True,
    pos_weight_clip    = 8.0,
    pos_weight_sqrt    = True,

    # Phase 2
    epochs_phase2      = 200,
    lr_phase2          = 1e-3,
    patience_phase2    = 10,
    dae_noise_std      = 0.02,

    # Phase 3 (XGB)
    xgb_estimators         = 5000,
    xgb_depth              = 10,
    xgb_lr                 = 0.1,
    xgb_lambda             = 0.3,
    xgb_n_jobs             = -1,
    xgb_use_scale_pos_weight = True,
    thr_min                = 0.05,
    thr_max                = 0.95,
    thr_steps              = 101,
    print_thresholds       = True,

    # 기타
    num_workers        = 2 if IS_KAGGLE else 4,
    resume             = False,
    save_phase3        = False,   # ablation 결과가 본실험 pkl 덮어쓰지 않도록
    current_test_sid   = None,
    ablation           = "full",
)


def _make_args(overrides: dict) -> types.SimpleNamespace:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def _print_result(label: str, acc: float, acc_per_task: np.ndarray) -> None:
    print(f"\n{'─'*50}")
    print(f"[{label}]")
    for name, a in zip(TASK_NAMES, acc_per_task):
        print(f"  {name:<8}: {float(a):.4f}")
    print(f"  {'Mean':<8}: {acc:.4f}")
    print(f"{'─'*50}")


def run_ablation(loso_sid: int = 0, config_overrides: dict = None):
    """
    5가지 ablation 조건을 순서대로 실행합니다.

      1. No P1+P2    : raw merged feature → XGB           (베이스라인)
      2. No P1       : raw merged feature → DAE → XGB     (P2만 있을 때)
      3. No P2+P3    : P1 classifier head 직접 평가        (P1만 있을 때)
      4. No P2       : P1 latent → XGB                    (P1+P3)
      5. Full        : P1 → DAE → XGB                     (전체)

    Parameters
    ----------
    loso_sid : int
        테스트 피험자 번호 (default: 0)
    config_overrides : dict, optional
        DEFAULT_CONFIG 값을 덮어쓸 항목
    """
    overrides = config_overrides or {}
    overrides["loso_sid"] = loso_sid

    args_full  = _make_args({**overrides, "ablation": "full"})
    args_no_p2 = _make_args({**overrides, "ablation": "no_p2", "resume": True})

    os.makedirs(args_full.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ablation target: SID={loso_sid}")

    dtcwt_cfg = DTCWTConfig()

    # ── 데이터 로드 (한 번만) ─────────────────────────────────────────────────
    print("\n[Data] Loading LOSO fold...")
    tr_ds = va_ds = te_ds = None
    for _tr, _va, _te, _, _sid in loso_splits(
        args_full.raw_data_dir,
        seed=args_full.seed,
        verbose=False,
        augment_train=args_full.augment,
        dtcwt_cfg=dtcwt_cfg,
        channel_mode=args_full.channel_mode,
        feature_mode=args_full.feature_mode,
        ccv_repr=args_full.ccv_repr,
        ccv_reject_alpha=args_full.ccv_reject_alpha,
        cov_bounded_scale=args_full.cov_bounded_scale,
        apply_ica=args_full.apply_ica,
        ica_method=args_full.ica_method,
        ica_n_components=args_full.ica_n_components,
        ica_random_state=args_full.ica_random_state,
        ica_max_iter=args_full.ica_max_iter,
    ):
        if int(_sid) == int(loso_sid):
            tr_ds, va_ds, te_ds = _tr, _va, _te
            break

    if tr_ds is None:
        raise RuntimeError(f"SID {loso_sid} not found in dataset.")

    # n_channels 자동 설정
    sample = tr_ds[0]
    n_ch = int(sample["eeg"].shape[1])
    for a in (args_full, args_no_p2, args_no_p2p3):
        a.n_channels = n_ch
        a.current_test_sid = int(loso_sid)

    print(f"[Data] Train={len(tr_ds)}, Val={len(va_ds)}, Test={len(te_ds)}, C={n_ch}")

    tr_loader = DataLoader(tr_ds, batch_size=args_full.batch_size, shuffle=True,
                           collate_fn=tr_ds.collate_fn, num_workers=args_full.num_workers)
    va_loader = DataLoader(va_ds, batch_size=args_full.batch_size, shuffle=False,
                           collate_fn=va_ds.collate_fn, num_workers=args_full.num_workers)
    te_loader = DataLoader(te_ds, batch_size=args_full.batch_size, shuffle=False,
                           collate_fn=te_ds.collate_fn, num_workers=args_full.num_workers)

    subject_remap, n_subjects_fold = _build_subject_remap(tr_ds)

    results = {}

    def _get_dae_latent(dae_model, features):
        dae_model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(device)
            _, z = dae_model(x)
            return z.cpu().numpy()

    # ══════════════════════════════════════════════════════════════════════════
    # [공통] No P1 조건용: CNN+LSTM raw merged feature 추출 (랜덤 초기화)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Feature] Extracting raw merged features (No P1 conditions)...")
    raw_tr_feat, raw_tr_labels = extract_raw_merged_features(n_ch, tr_loader, device)
    raw_va_feat, raw_va_labels = extract_raw_merged_features(n_ch, va_loader, device)
    raw_te_feat, raw_te_labels = extract_raw_merged_features(n_ch, te_loader, device)

    # ══════════════════════════════════════════════════════════════════════════
    # Ablation 4: No P1+P2  (raw merged → XGB)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("Ablation: No P1+P2  (raw merged → XGB)")
    print("="*50)
    _, acc, apt, _ = train_phase3(
        raw_tr_feat, raw_tr_labels,
        raw_va_feat, raw_va_labels,
        raw_te_feat, raw_te_labels,
        args_no_p2,
    )
    results["No P1+P2 (raw→XGB)"] = (acc, apt)
    _print_result("No P1+P2 (raw→XGB)", acc, apt)

    # ══════════════════════════════════════════════════════════════════════════
    # Ablation 5: No P1  (raw merged → DAE → XGB)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("Ablation: No P1  (raw merged → DAE → XGB)")
    print("="*50)
    dae_no_p1 = train_phase2(raw_tr_feat, raw_va_feat, device, args_full)
    _, acc, apt, _ = train_phase3(
        _get_dae_latent(dae_no_p1, raw_tr_feat), raw_tr_labels,
        _get_dae_latent(dae_no_p1, raw_va_feat), raw_va_labels,
        _get_dae_latent(dae_no_p1, raw_te_feat), raw_te_labels,
        args_full,
    )
    results["No P1 (raw→DAE→XGB)"] = (acc, apt)
    _print_result("No P1 (raw→DAE→XGB)", acc, apt)

    # ══════════════════════════════════════════════════════════════════════════
    # [공통] Phase 1 학습 (P1 포함 조건들 공통 — 한 번만)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("Phase 1 학습 (공통)")
    print("="*50)
    model = train_phase1(tr_loader, va_loader, device, args_full, subject_remap, n_subjects_fold)

    # ══════════════════════════════════════════════════════════════════════════
    # Ablation 3: No P2+P3  (P1 classifier head 직접)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("Ablation: No P2+P3  (P1 head 직접)")
    print("="*50)
    acc, apt = evaluate_phase1_only(model, te_loader, device)
    results["No P2+P3 (P1 head)"] = (acc, apt)
    _print_result("No P2+P3 (P1 head)", acc, apt)

    # ══════════════════════════════════════════════════════════════════════════
    # [공통] P1 latent / merged feature 추출
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Feature] Extracting P1 latent features (No P2)...")
    train_latent, train_labels = extract_latent_features(model, tr_loader, device)
    val_latent,   val_labels   = extract_latent_features(model, va_loader, device)
    test_latent,  test_labels  = extract_latent_features(model, te_loader, device)

    print("[Feature] Extracting P1 merged features (Full)...")
    train_feat, _ = extract_merged_features(model, tr_loader, device)
    val_feat,   _ = extract_merged_features(model, va_loader, device)
    test_feat,  _ = extract_merged_features(model, te_loader, device)

    # ══════════════════════════════════════════════════════════════════════════
    # Ablation 2: No P2  (P1 latent → XGB)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("Ablation: No P2  (P1 latent → XGB)")
    print("="*50)
    _, acc, apt, _ = train_phase3(
        train_latent, train_labels,
        val_latent,   val_labels,
        test_latent,  test_labels,
        args_no_p2,
    )
    results["No P2 (P1→XGB)"] = (acc, apt)
    _print_result("No P2 (P1→XGB)", acc, apt)

    # ══════════════════════════════════════════════════════════════════════════
    # Full: P1 → P2 (DAE) → P3 (XGB)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("Full Pipeline: P1 → P2 (DAE) → P3 (XGB)")
    print("="*50)
    dae_full = train_phase2(train_feat, val_feat, device, args_full)
    _, acc, apt, _ = train_phase3(
        _get_dae_latent(dae_full, train_feat), train_labels,
        _get_dae_latent(dae_full, val_feat),   val_labels,
        _get_dae_latent(dae_full, test_feat),  test_labels,
        args_full,
    )
    results["Full (P1+P2+P3)"] = (acc, apt)
    _print_result("Full (P1+P2+P3)", acc, apt)

    # ══════════════════════════════════════════════════════════════════════════
    # 최종 비교표
    # ══════════════════════════════════════════════════════════════════════════
    df = _print_summary(results)
    return results, df


def _print_summary(results: dict) -> None:
    rows = []
    for label, (acc, apt) in results.items():
        row = {"Condition": label}
        for name, a in zip(TASK_NAMES, apt):
            row[name] = round(float(a), 4)
        row["Mean"] = round(float(acc), 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Condition")

    # Best per column 강조 (터미널용)
    print("\n" + "="*60)
    print("Ablation Summary")
    print("="*60)
    print(df.to_string())
    print("="*60)

    # Best condition
    best = df["Mean"].idxmax()
    print(f"\n✓ Best: {best}  (Mean={df.loc[best, 'Mean']:.4f})")

    # 혹시 IPython 환경이면 styled DataFrame도 표시
    try:
        from IPython.display import display
        styled = (
            df.style
            .highlight_max(axis=0, color="lightgreen")
            .format("{:.4f}")
            .set_caption("Ablation Study Results")
        )
        display(styled)
    except Exception:
        pass

    return df


if __name__ == "__main__":
    # ── 노트북에서 실행할 때는 아래처럼 sys.argv 설정 후 호출 ─────────────────
    # import sys
    # sys.argv = ["ablation_study.py"]   # 추가 인자 필요 시 여기에
    # results, df = run_ablation(loso_sid=0)
    # ─────────────────────────────────────────────────────────────────────────
    results, df = run_ablation(loso_sid=0)
