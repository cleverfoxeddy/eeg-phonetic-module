# 목적: P1 (SpatialCNN+LSTM+GRL) + P2 (DAE) LOSO 체크포인트 생성
#   - phase1_best_sid{N}.pt  : SpatialCNN+LSTM 인코더
#   - phase2_best_sid{N}.pt  : DAE (1152→32→1152)
#   둘 다 있어야 실제 DAE latent (32d) 추출 가능
#
# 캐글 셀:
#   exec(open("/kaggle/input/datasets/hdwmmmlc/codeupload/train_p1p2_checkpoints.py").read())
#   run_p1p2_checkpoints()          # 전체 14 SID
#   run_p1p2_checkpoints(sids=[0])  # 특정 SID만
#
# 이후 t-SNE:
#   extract_tsne_kaggle.py 의 CKPT_DIR = "/kaggle/working/checkpoints_p1p2"
#
# resume=True (기본): 이미 존재하는 체크포인트 스킵 → 중간 재시작 가능

import os, sys, random, copy, types
import numpy as np

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle/input")

for _p in [
    "/kaggle/input/datasets/hdwmmmlc/codeupload",
    "/kaggle/working",
    os.getcwd(),
]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from kara_one_dataset import loso_splits
from model_manifold import EEGPhonologicalManifoldNet
from train_stage2_manifold import (
    DTCWTConfig,
    _KAGGLE_DATA_DIR,
    _build_subject_remap,
    _linear_warmup,
    _estimate_task_pos_weight,
    extract_merged_features,   # model → 1152d
)

# ── 설정 ───────────────────────────────────────────────────────────────────────
DATA_DIR     = _KAGGLE_DATA_DIR if IS_KAGGLE else "/data/kara_one"
CKPT_DIR     = "/kaggle/working/checkpoints_p1p2"
CHANNEL_MODE = "all"   # "all"=64ch | "paper10"=10ch
SEED         = 42

CFG = dict(
    seed               = SEED,
    latent_dim         = 32,
    batch_size         = 16,
    num_workers        = 2 if IS_KAGGLE else 4,
    # Phase 1
    epochs_phase1      = 80,
    lr_phase1          = 1e-3,
    wd                 = 0.0,
    patience_phase1    = 8,
    adv_weight         = 0.0,    # GRL 끄고 순수 P1만. 켜려면 0.5로 변경
    grl_lambda         = 0.5,
    adv_warmup_epochs  = 10,
    adv_val_weight     = 0.0,
    use_task_pos_weight= False,
    pos_weight_clip    = 10.0,
    pos_weight_sqrt    = False,
    augment            = False,
    augment_start      = 0,
    # Phase 2 (DAE)
    epochs_phase2      = 200,
    lr_phase2          = 1e-3,
    patience_phase2    = 10,
    dae_noise_std      = 0.02,
    # feature
    channel_mode       = CHANNEL_MODE,
    feature_mode       = "raw",
    ccv_repr           = "cov_bounded",
    ccv_reject_alpha   = 0.10,
    cov_bounded_scale  = 0.10,
    apply_ica          = False,
    ica_method         = "fastica",
    ica_n_components   = None,
    ica_random_state   = 97,
    ica_max_iter       = 512,
    # placeholder
    n_channels         = 64,
    current_test_sid   = None,
    ckpt_dir           = CKPT_DIR,
    resume             = True,
)


def _make_args(overrides=None):
    cfg = copy.deepcopy(CFG)
    if overrides:
        cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


# ── DeepAutoencoder (train_stage2_manifold.DeepAutoencoder 와 동일) ─────────────
class _DAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 128),       nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 512),        nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ── Phase 1 ────────────────────────────────────────────────────────────────────
def _train_phase1(tr_ds, va_ds, args, device):
    best_path = os.path.join(args.ckpt_dir, f"phase1_best_sid{args.current_test_sid}.pt")

    if args.resume and os.path.isfile(best_path):
        print(f"    [P1 skip] {best_path}")
        model = EEGPhonologicalManifoldNet(
            n_channels=args.n_channels, n_subjects=13,
            n_tasks=5, latent_dim=args.latent_dim, grl_lambda=0.0,
        ).to(device)
        model.load_state_dict(torch.load(best_path, map_location=device))
        return model

    subject_remap, n_subj = _build_subject_remap(tr_ds)
    tr_ldr = DataLoader(tr_ds, args.batch_size, shuffle=True,
                        collate_fn=tr_ds.collate_fn, num_workers=args.num_workers)
    va_ldr = DataLoader(va_ds, args.batch_size, shuffle=False,
                        collate_fn=va_ds.collate_fn, num_workers=args.num_workers)

    model = EEGPhonologicalManifoldNet(
        n_channels=args.n_channels, n_subjects=int(n_subj),
        n_tasks=5, latent_dim=args.latent_dim, grl_lambda=0.0,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr_phase1, weight_decay=args.wd)

    pw = None
    if args.use_task_pos_weight:
        pw = _estimate_task_pos_weight(tr_ldr, device)
        if pw is not None:
            if args.pos_weight_sqrt: pw = torch.sqrt(pw)
            if args.pos_weight_clip > 0: pw = pw.clamp(1.0, float(args.pos_weight_clip))
    crit_p = nn.BCEWithLogitsLoss(pos_weight=pw) if pw is not None else nn.BCEWithLogitsLoss()
    crit_s = nn.CrossEntropyLoss(ignore_index=-1)

    best_val, patience = float("inf"), 0

    for ep in range(1, args.epochs_phase1 + 1):
        grl = _linear_warmup(ep, args.adv_warmup_epochs, args.grl_lambda) \
              if args.adv_weight > 0 else 0.0
        try:    model.set_grl_lambda(grl)
        except: model.grl_lambda = float(grl)

        model.train()
        for batch in tr_ldr:
            eeg  = batch["eeg"].to(device)
            corr = batch["corr"].to(device)
            mask = batch.get("corr_mask", torch.ones_like(batch["corr"])).to(device)
            attn = batch["attention_mask"].to(device)
            tgt  = batch["phon"].to(device).float()
            if args.augment and ep >= args.augment_start:
                eeg = eeg + 0.05 * torch.randn_like(eeg)
                if torch.rand(1).item() > 0.5:
                    eeg = eeg * (0.9 + 0.2 * torch.rand(1).item())
            sid_t = torch.tensor(
                [subject_remap[int(s)] for s in batch["subject_id"].cpu().view(-1).tolist()],
                device=device, dtype=torch.long,
            )
            logits, _, s_logits, _, _ = model(eeg, corr, mask, attn)
            loss = crit_p(logits, tgt)
            if args.adv_weight > 0:
                loss = loss + args.adv_weight * crit_s(s_logits, sid_t)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        va_loss, nv = 0.0, 0
        with torch.no_grad():
            for batch in va_ldr:
                eeg  = batch["eeg"].to(device)
                corr = batch["corr"].to(device)
                mask = batch.get("corr_mask", torch.ones_like(batch["corr"])).to(device)
                attn = batch["attention_mask"].to(device)
                tgt  = batch["phon"].to(device).float()
                logits, _, _, _, _ = model(eeg, corr, mask, attn)
                va_loss += crit_p(logits, tgt).item() * eeg.size(0)
                nv += eeg.size(0)
        va_loss /= max(1, nv)

        if ep % 10 == 0 or ep == 1:
            print(f"    [P1] ep={ep:3d}  val={va_loss:.4f}")

        if va_loss < best_val:
            best_val, patience = va_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if args.patience_phase1 > 0 and patience >= args.patience_phase1:
                print(f"    [P1] early stop @ ep={ep}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"    [P1] ✓ saved → {best_path}  (best_val={best_val:.4f})")
    return model


# ── Phase 2 (DAE) ──────────────────────────────────────────────────────────────
def _train_phase2(tr_feat, va_feat, args, device):
    best_path = os.path.join(args.ckpt_dir, f"phase2_best_sid{args.current_test_sid}.pt")
    input_dim = tr_feat.shape[1]  # 1152

    if args.resume and os.path.isfile(best_path):
        print(f"    [P2 skip] {best_path}")
        dae = _DAE(input_dim, args.latent_dim).to(device)
        dae.load_state_dict(torch.load(best_path, map_location=device))
        return dae

    tr_ldr = DataLoader(
        TensorDataset(torch.from_numpy(tr_feat).float()),
        batch_size=args.batch_size, shuffle=True,
    )
    va_ldr = DataLoader(
        TensorDataset(torch.from_numpy(va_feat).float()),
        batch_size=args.batch_size, shuffle=False,
    )

    dae = _DAE(input_dim, args.latent_dim).to(device)
    opt = optim.Adam(dae.parameters(), lr=args.lr_phase2, weight_decay=args.wd)
    crit = nn.MSELoss()

    best_val, patience = float("inf"), 0

    for ep in range(1, args.epochs_phase2 + 1):
        dae.train()
        for (x,) in tr_ldr:
            x = x.to(device)
            noise = torch.randn_like(x) * (args.dae_noise_std * x.std(dim=1, keepdim=True))
            recon, _ = dae(x + noise)
            loss = crit(recon, x)
            opt.zero_grad(); loss.backward(); opt.step()

        dae.eval()
        va_loss, nv = 0.0, 0
        with torch.no_grad():
            for (x,) in va_ldr:
                x = x.to(device)
                recon, _ = dae(x)
                va_loss += crit(recon, x).item() * x.size(0)
                nv += x.size(0)
        va_loss /= max(1, nv)

        if ep % 20 == 0 or ep == 1:
            print(f"    [P2] ep={ep:3d}  val_mse={va_loss:.6f}")

        if va_loss < best_val:
            best_val, patience = va_loss, 0
            torch.save(dae.state_dict(), best_path)
        else:
            patience += 1
            if args.patience_phase2 > 0 and patience >= args.patience_phase2:
                print(f"    [P2] early stop @ ep={ep}")
                break

    dae.load_state_dict(torch.load(best_path, map_location=device))
    print(f"    [P2] ✓ saved → {best_path}  (best_val={best_val:.6f})")
    return dae


# ── main ───────────────────────────────────────────────────────────────────────
def run_p1p2_checkpoints(sids=None, channel_mode=CHANNEL_MODE, seed=SEED):
    """
    Parameters
    ----------
    sids : list[int] | None  — None이면 0~13 전체
    channel_mode : "all" | "paper10"
    seed : int
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    if sids is None:
        sids = list(range(14))

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtcwt_cfg = DTCWTConfig()

    print(f"{'='*60}")
    print(f"P1 + P2 (DAE) Checkpoint Generation")
    print(f"  channel_mode : {channel_mode}")
    print(f"  sids         : {sids}")
    print(f"  ckpt_dir     : {CKPT_DIR}")
    print(f"  device       : {device}")
    print(f"  adv_weight   : {CFG['adv_weight']}  (0=GRL off)")
    print(f"{'='*60}\n")

    for tr_ds, va_ds, te_ds, _, test_sid in loso_splits(
        DATA_DIR, seed=seed, verbose=False,
        dtcwt_cfg=dtcwt_cfg, channel_mode=channel_mode,
        feature_mode=CFG["feature_mode"], ccv_repr=CFG["ccv_repr"],
        ccv_reject_alpha=CFG["ccv_reject_alpha"],
        cov_bounded_scale=CFG["cov_bounded_scale"],
        apply_ica=CFG["apply_ica"], ica_method=CFG["ica_method"],
        ica_n_components=CFG["ica_n_components"],
        ica_random_state=CFG["ica_random_state"],
        ica_max_iter=CFG["ica_max_iter"],
    ):
        if int(test_sid) not in sids:
            continue

        n_ch = int(tr_ds[0]["eeg"].shape[1])
        args = _make_args({
            "n_channels"      : n_ch,
            "current_test_sid": int(test_sid),
            "channel_mode"    : channel_mode,
            "seed"            : seed,
        })

        print(f"── SID {test_sid}  (C={n_ch}, tr={len(tr_ds)}, va={len(va_ds)}, te={len(te_ds)}) ──")

        # Phase 1
        p1_model = _train_phase1(tr_ds, va_ds, args, device)

        # Phase 1로 1152d feature 추출
        tr_ldr = DataLoader(tr_ds, args.batch_size, shuffle=False,
                            collate_fn=tr_ds.collate_fn, num_workers=args.num_workers)
        va_ldr = DataLoader(va_ds, args.batch_size, shuffle=False,
                            collate_fn=va_ds.collate_fn, num_workers=args.num_workers)

        tr_feat, _ = extract_merged_features(p1_model, tr_ldr, device)
        va_feat, _ = extract_merged_features(p1_model, va_ldr, device)
        print(f"    [feat] tr={tr_feat.shape}, va={va_feat.shape}")

        # Phase 2 (DAE)
        _train_phase2(tr_feat, va_feat, args, device)

        print()

    print(f"{'='*60}")
    print(f"Done.")
    print(f"  phase1_best_sid{{N}}.pt  ← P1 encoder")
    print(f"  phase2_best_sid{{N}}.pt  ← DAE (1152→32→1152)")
    print(f"\n다음 단계: extract_tsne_kaggle.py 에서")
    print(f"  CKPT_DIR = '{CKPT_DIR}'  으로 설정 후 실행")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_p1p2_checkpoints()