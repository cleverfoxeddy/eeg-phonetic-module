# train_stage2_manifold.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import pickle
from sklearn.multioutput import MultiOutputClassifier  # kept for potential future use
from torch.utils.data import DataLoader, TensorDataset
import argparse

from kara_one_dataset import (
    create_random_splits,
    create_trial_random_splits,
    loso_splits,
    DTCWTConfig,
)
from precompute_dtcwt import PrecomputedDataset
from model_manifold import EEGPhonologicalManifoldNet

TASK_NAMES = ["vowel", "nasal", "bilabial", "iy", "uw"]


def _linear_warmup(epoch_idx: int, warmup_epochs: int, target: float) -> float:
    """GRL lambda를 epoch 1 기준으로 0 → target 까지 선형 증가."""
    if warmup_epochs <= 0:
        return float(target)
    t = min(1.0, max(0.0, epoch_idx / float(warmup_epochs)))
    return float(target) * t


def _collect_subject_ids(ds, max_scan: int = 200000):
    """
    Robustly collect unique subject ids from dataset without assuming attributes.
    Uses __getitem__ scan (capped).
    """
    uniq = set()
    n = min(len(ds), max_scan)
    for i in range(n):
        it = ds[i]
        sid = int(it["subject_id"].item()) if torch.is_tensor(it["subject_id"]) else int(it["subject_id"])
        uniq.add(sid)
    return sorted(uniq)


def _build_subject_remap(train_ds):
    """
    Build remap based on train subjects only.
    For adversarial invariance we only need remap over subjects actually seen during training.
    """
    s_tr = _collect_subject_ids(train_ds)
    remap = {sid: i for i, sid in enumerate(s_tr)}
    return remap, len(s_tr)


def _estimate_task_pos_weight(train_loader, device, eps: float = 1e-6):
    """pos_weight[t] = #neg / #pos for each task. Combats label imbalance."""
    pos = None
    tot = 0
    for batch in train_loader:
        y = batch["phon"].to(device).float()
        if pos is None:
            pos = torch.zeros(y.size(1), device=device)
        pos += y.sum(dim=0)
        tot += y.size(0)
    if pos is None:
        return None
    neg = (float(tot) - pos).clamp(min=eps)
    pos = pos.clamp(min=eps)
    return (neg / pos).detach()


# ---------------- Phase 1 ----------------
def train_phase1(train_loader, val_loader, device, args, subject_remap, n_subjects_fold):
    model = EEGPhonologicalManifoldNet(
        n_channels=args.n_channels,
        n_subjects=int(n_subjects_fold),
        n_tasks=5,
        latent_dim=args.latent_dim,
        grl_lambda=0.0,  # will be scheduled per-epoch via set_grl_lambda()
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_phase1, weight_decay=args.wd)
    # Task-wise pos_weight for label imbalance
    pos_weight = None
    if args.use_task_pos_weight:
        pos_weight = _estimate_task_pos_weight(train_loader, device=device)
        if pos_weight is not None:
            if args.pos_weight_sqrt:
                pos_weight = torch.sqrt(pos_weight)
            if args.pos_weight_clip > 0:
                pos_weight = pos_weight.clamp(min=1.0, max=float(args.pos_weight_clip))
            print(f"[Phase1] pos_weight={pos_weight.cpu().numpy().round(3).tolist()}")
    criterion_phon = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    # allow ignoring unseen subjects in val (mapped to -1)
    criterion_subject = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_loss, patience_counter = float("inf"), 0
    # NOTE: In LOSO-CV we must not overwrite checkpoints across folds.
    sid = getattr(args, "current_test_sid", None)
    if sid is None:
        best_path = os.path.join(args.ckpt_dir, "phase1_best.pt")
    else:
        best_path = os.path.join(args.ckpt_dir, f"phase1_best_sid{int(sid)}.pt")

    for epoch in range(1, args.epochs_phase1 + 1):
        # epoch-based augmentation: active only from augment_start epoch onward
        aug_active = getattr(args, "augment", False) and (epoch >= getattr(args, "augment_start", 1))
        if epoch == getattr(args, "augment_start", 1) and aug_active:
            print(f"  [Augment] Epoch {epoch}: in-loop augmentation activated.")

        # GRL lambda warmup
        grl_now = _linear_warmup(epoch, args.adv_warmup_epochs, args.grl_lambda) if args.adv_weight > 0 else 0.0
        try:
            model.set_grl_lambda(grl_now)
        except Exception:
            model.grl_lambda = float(grl_now)

        model.train()
        train_loss, n_samples = 0.0, 0
        train_adv_loss, train_adv_correct = 0.0, 0

        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            corr = batch["corr"].to(device)
            corr_mask = batch.get("corr_mask", torch.ones_like(batch["corr"])).to(device)
            attn = batch["attention_mask"].to(device)
            targets = batch["phon"].to(device).float()

            # in-loop augmentation (epoch-gated by augment_start)
            if aug_active:
                # Gaussian noise on EEG time series
                eeg = eeg + 0.05 * torch.randn_like(eeg)
                # random time-axis scaling (±10%)
                if torch.rand(1).item() > 0.5:
                    scale = 0.9 + 0.2 * torch.rand(1).item()
                    eeg = eeg * scale

            # remap subject ids to 0..K-1 for this fold
            raw_sid = batch["subject_id"].cpu().view(-1).tolist()
            mapped = [subject_remap[int(s)] for s in raw_sid]
            subject_id = torch.tensor(mapped, device=device, dtype=torch.long)

            logits, _, subject_logits, _, _ = model(eeg, corr, corr_mask, attn)
            phon_loss = criterion_phon(logits, targets)

            if args.adv_weight > 0:
                adv_loss = criterion_subject(subject_logits, subject_id)
                loss = phon_loss + float(args.adv_weight) * adv_loss
                with torch.no_grad():
                    pred_sid = subject_logits.argmax(dim=1)
                    train_adv_correct += (pred_sid == subject_id).sum().item()
                train_adv_loss += adv_loss.item() * eeg.size(0)
            else:
                loss = phon_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * eeg.size(0)
            n_samples += eeg.size(0)

        train_loss /= max(1, n_samples)
        train_adv_loss /= max(1, n_samples)
        train_adv_acc = train_adv_correct / max(1, n_samples)

        model.eval()
        val_loss, val_n = 0.0, 0
        val_adv_loss_sum, val_adv_correct_sum = 0.0, 0.0
        val_adv_seen_n = 0
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch["eeg"].to(device)
                corr = batch["corr"].to(device)
                corr_mask = batch.get("corr_mask", torch.ones_like(batch["corr"])).to(device)
                attn = batch["attention_mask"].to(device)
                targets = batch["phon"].to(device).float()

                raw_sid = batch["subject_id"].cpu().view(-1).tolist()
                # map unseen subjects to -1 (ignored by CE)
                mapped = [subject_remap.get(int(s), -1) for s in raw_sid]
                subject_id = torch.tensor(mapped, device=device, dtype=torch.long)

                # keep same grl on eval pass (not critical, but consistent)
                grl_now = _linear_warmup(epoch, args.adv_warmup_epochs, args.grl_lambda) if args.adv_weight > 0 else 0.0
                try:
                    model.set_grl_lambda(grl_now)
                except Exception:
                    model.grl_lambda = float(grl_now)

                logits, _, subject_logits, _, _ = model(eeg, corr, corr_mask, attn)
                phon_loss = criterion_phon(logits, targets)
                total = phon_loss

                # compute adv only on seen subjects in this fold (subject_id != -1)
                seen_mask = (subject_id != -1)
                if seen_mask.any() and args.adv_weight > 0 and args.adv_val_weight > 0:
                    seen_logits = subject_logits[seen_mask]
                    seen_targets = subject_id[seen_mask]
                    adv_loss = criterion_subject(seen_logits, seen_targets)
                    total = phon_loss + float(args.adv_val_weight) * adv_loss

                    seen_n = int(seen_targets.numel())
                    val_adv_seen_n += seen_n
                    val_adv_loss_sum += adv_loss.item() * seen_n

                    pred_sid = seen_logits.argmax(dim=1)
                    val_adv_correct_sum += float((pred_sid == seen_targets).sum().item())

                val_loss += total.item() * eeg.size(0)
                val_n += eeg.size(0)

        val_loss /= max(1, val_n)

        if val_adv_seen_n > 0:
            val_adv_loss = val_adv_loss_sum / float(val_adv_seen_n)
            val_adv_acc = val_adv_correct_sum / float(val_adv_seen_n)
        else:
            val_adv_loss = 0.0
            val_adv_acc = 0.0

        if args.adv_weight > 0:
            print(
                f"Phase1 Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} (adv_ce={train_adv_loss:.4f}, adv_acc={train_adv_acc:.3f}, grl={grl_now:.3f}) | "
                f"Val Loss: {val_loss:.4f} (adv_ce={val_adv_loss:.4f}, adv_acc={val_adv_acc:.3f}, seen={val_adv_seen_n})"
            )
        else:
            print(f"Phase1 Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  --> Best model saved (epoch {epoch})")
        else:
            patience_counter += 1
            if args.patience_phase1 > 0 and patience_counter >= args.patience_phase1:
                print(f"  Early stopping after {epoch} epochs.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


def extract_merged_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            latent, merged, merged_hat = model.encode(
                batch["eeg"].to(device),
                batch["corr"].to(device),
                batch.get("corr_mask", torch.ones_like(batch["corr"])).to(device),
                batch["attention_mask"].to(device),
            )
            feats.append(merged.cpu().numpy())
            labels.append(batch["phon"].cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels)


def extract_latent_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            latent, merged, merged_hat = model.encode(
                batch["eeg"].to(device),
                batch["corr"].to(device),
                batch.get("corr_mask", torch.ones_like(batch["corr"])).to(device),
                batch["attention_mask"].to(device),
            )
            feats.append(latent.cpu().numpy())
            labels.append(batch["phon"].cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ---------------- Phase 2 (DAE) ----------------
class DeepAutoencoder(nn.Module):
    """
    Keep it simple; if you want strict paper Table-1 activation schedule,
    we can swap this to exact spec later.
    """

    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_phase2(train_features, val_features, device, args):
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_features).float()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_features).float()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = DeepAutoencoder(input_dim=train_features.shape[1], latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_phase2, weight_decay=args.wd)
    criterion = nn.MSELoss()

    best_val_loss, patience_counter = float("inf"), 0
    # NOTE: In LOSO-CV we must not overwrite checkpoints across folds.
    sid = getattr(args, "current_test_sid", None)
    if sid is None:
        best_path = os.path.join(args.ckpt_dir, "phase2_best.pt")
    else:
        best_path = os.path.join(args.ckpt_dir, f"phase2_best_sid{int(sid)}.pt")

    # ── Phase2 resume: skip training if checkpoint already exists ──────────────
    if args.resume and os.path.isfile(best_path):
        print(f"  [Phase2] Resuming from existing checkpoint: {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=device))
        return model
    # ──────────────────────────────────────────────────────────────────────────

    for epoch in range(1, args.epochs_phase2 + 1):
        model.train()
        for (x_clean,) in train_loader:
            x_clean = x_clean.to(device)
            noise = torch.randn_like(x_clean) * (args.dae_noise_std * x_clean.std(dim=1, keepdim=True))
            x_noisy = x_clean + noise
            recon, _ = model(x_noisy)
            loss = criterion(recon, x_clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for (x_clean,) in val_loader:
                x_clean = x_clean.to(device)
                recon, _ = model(x_clean)
                val_loss += criterion(recon, x_clean).item() * x_clean.size(0)
                n_val += x_clean.size(0)

        val_loss /= max(1, n_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Phase2 Epoch {epoch:4d} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if args.patience_phase2 > 0 and patience_counter >= args.patience_phase2:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


# ---------------- Phase 3 (XGB) ----------------
def _tune_thresholds_on_val(models, val_x, val_y, thr_min, thr_max, thr_steps):
    """Task-wise threshold tuning on validation set to maximize per-task accuracy."""
    grid = np.linspace(thr_min, thr_max, thr_steps)
    thresholds = []
    for t, m in enumerate(models):
        p = m.predict_proba(val_x)[:, 1]
        y = val_y[:, t].astype(int)
        best_thr, best_acc = 0.5, -1.0
        for thr in grid:
            acc = ((p >= thr).astype(int) == y).mean()
            if acc > best_acc:
                best_acc = acc
                best_thr = float(thr)
        thresholds.append(best_thr)
    return np.array(thresholds, dtype=np.float32)


def _phase3_ckpt_path(args):
    """Return the pickle path for Phase3 XGB checkpoint."""
    sid = getattr(args, "current_test_sid", None)
    if sid is None:
        return os.path.join(args.ckpt_dir, "phase3_xgb.pkl")
    return os.path.join(args.ckpt_dir, f"phase3_xgb_sid{int(sid)}.pkl")


def save_phase3(models, thresholds, args):
    """Persist XGB models + tuned thresholds to disk via pickle."""
    path = _phase3_ckpt_path(args)
    payload = {"models": models, "thresholds": thresholds}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  [Phase3] Checkpoint saved → {path}")


def load_phase3(args):
    """Load XGB models + thresholds from disk. Returns (models, thresholds) or None."""
    path = _phase3_ckpt_path(args)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"  [Phase3] Checkpoint loaded ← {path}")
    return payload["models"], payload["thresholds"]


def train_phase3(train_latent, train_labels, val_latent, val_labels, test_latent, test_labels, args):
    """Per-task XGBoost with optional scale_pos_weight + val-tuned thresholds.

    Checkpoint behaviour
    --------------------
    * ``--resume``:  if a Phase3 checkpoint already exists for this fold,
      skip training entirely and evaluate straight from the saved models.
    * ``--save_phase3`` (default True):  after training, persist the fitted
      models + tuned thresholds so they can be resumed later.
    """

    # ── Phase3 resume ─────────────────────────────────────────────────────────
    if args.resume:
        cached = load_phase3(args)
        if cached is not None:
            models, thresholds = cached
            pred = np.stack(
                [(m.predict_proba(test_latent)[:, 1] >= thresholds[t]).astype(int)
                 for t, m in enumerate(models)], axis=1
            )
            acc_per_task = (pred == test_labels.astype(int)).mean(axis=0)
            return models, float(acc_per_task.mean()), acc_per_task, thresholds
    # ──────────────────────────────────────────────────────────────────────────

    models = []
    for t in range(train_labels.shape[1]):
        y_tr = train_labels[:, t].astype(int)
        pos = max(1, int(y_tr.sum()))
        neg = max(1, int(len(y_tr) - pos))
        spw = (neg / float(pos)) if args.xgb_use_scale_pos_weight else 1.0
        m = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=args.xgb_estimators,
            max_depth=args.xgb_depth,
            learning_rate=args.xgb_lr,
            subsample=0.8,
            colsample_bytree=0.4,
            reg_lambda=args.xgb_lambda,
            random_state=args.seed,
            n_jobs=args.xgb_n_jobs,
            eval_metric="logloss",
            scale_pos_weight=spw,
            verbosity=0,
        )
        m.fit(train_latent, y_tr)
        models.append(m)

    thresholds = _tune_thresholds_on_val(
        models, val_latent, val_labels,
        args.thr_min, args.thr_max, args.thr_steps,
    )
    if args.print_thresholds:
        print(f"[Phase3] tuned thresholds={thresholds.round(3).tolist()}")

    # ── save checkpoint ────────────────────────────────────────────────────────
    if args.save_phase3:
        save_phase3(models, thresholds, args)
    # ──────────────────────────────────────────────────────────────────────────

    pred = np.stack(
        [(m.predict_proba(test_latent)[:, 1] >= thresholds[t]).astype(int)
         for t, m in enumerate(models)], axis=1
    )
    acc_per_task = (pred == test_labels.astype(int)).mean(axis=0)
    return models, float(acc_per_task.mean()), acc_per_task, thresholds


# ---------------- pipeline ----------------
def run_pipeline(tr_ds, va_ds, te_ds, device, args):
    # fold-specific subject remap for adversarial loss (TRAIN subjects only)
    # In current LOSO split, val subjects are disjoint => do not include them in remap.
    subject_remap, n_subjects_fold = _build_subject_remap(tr_ds)
    if args.adv_weight > 0:
        print(f"[Fold] adversarial subject classes (TRAIN only): K={n_subjects_fold} | ids={list(subject_remap.keys())[:10]}{'...' if n_subjects_fold>10 else ''}")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=tr_ds.collate_fn, num_workers=args.num_workers)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=va_ds.collate_fn, num_workers=args.num_workers)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=te_ds.collate_fn, num_workers=args.num_workers)

    model = train_phase1(tr_loader, va_loader, device, args, subject_remap, n_subjects_fold)

    if args.epochs_phase2 > 0:
        train_feat, train_labels = extract_merged_features(model, tr_loader, device)
        val_feat, val_labels = extract_merged_features(model, va_loader, device)
        test_feat, test_labels = extract_merged_features(model, te_loader, device)

        dae = train_phase2(train_feat, val_feat, device, args)

        def get_latent(dae_model, features):
            dae_model.eval()
            with torch.no_grad():
                x = torch.from_numpy(features).float().to(device)
                _, z = dae_model(x)
                return z.cpu().numpy()

        train_latent = get_latent(dae, train_feat)
        val_latent   = get_latent(dae, val_feat)
        test_latent  = get_latent(dae, test_feat)
    else:
        train_latent, train_labels = extract_latent_features(model, tr_loader, device)
        val_latent,   val_labels   = extract_latent_features(model, va_loader, device)
        test_latent,  test_labels  = extract_latent_features(model, te_loader, device)

    _, test_acc, acc_per_task, _ = train_phase3(
        train_latent, train_labels,
        val_latent, val_labels,
        test_latent, test_labels, args,
    )
    return test_acc, acc_per_task


def main():
    parser = argparse.ArgumentParser()

    # data modes
    parser.add_argument("--data_dir", type=str, default="./precomputed")
    parser.add_argument("--raw_data_dir", type=str, default="/data/kara_one")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints_hierarchical_paper")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", type=str, default="loso_cv", choices=["precomputed", "subject", "trial", "loso_cv"])
    parser.add_argument("--loso_sid", type=int, default=None)

    # ── checkpoint / resume ────────────────────────────────────────────────────
    parser.add_argument("--resume", action="store_true",
                        help="If set, skip Phase2/Phase3 training when a checkpoint already exists "
                             "for the current fold and load from disk instead.")
    parser.add_argument("--save_phase3", action="store_true", default=True,
                        help="Persist Phase3 XGB models + thresholds after training (default: True). "
                             "Disable with --no_save_phase3.")
    parser.add_argument("--no_save_phase3", dest="save_phase3", action="store_false",
                        help="Do NOT save Phase3 XGB checkpoint.")
    # ──────────────────────────────────────────────────────────────────────────

    # paper-aligned feature knobs
    parser.add_argument("--channel_mode", type=str, default="all", choices=["paper10", "all"])
    parser.add_argument("--feature_mode", type=str, default="raw", choices=["dtcwt_beta", "raw"])
    parser.add_argument("--ccv_repr", type=str, default="cov_bounded", choices=["corr", "cov_bounded"])
    parser.add_argument("--ccv_reject_alpha", type=float, default=0.10)
    parser.add_argument("--cov_bounded_scale", type=float, default=0.10)

    # ICA knobs
    parser.add_argument("--apply_ica", action="store_true", help="Enable best-effort ICA ocular removal.")
    parser.add_argument("--ica_method", type=str, default="fastica")
    parser.add_argument("--ica_n_components", type=int, default=0, help="0 means None")
    parser.add_argument("--ica_random_state", type=int, default=97)
    parser.add_argument("--ica_max_iter", type=int, default=512)

    # adversarial (subject-invariant) knobs
    parser.add_argument("--adv_weight", type=float, default=0.0,
                        help="Subject CE loss weight (added to phon loss). 0 = disable adversarial.")
    parser.add_argument("--grl_lambda", type=float, default=0.5,
                        help="Target GRL lambda. Ramped from 0→target over adv_warmup_epochs.")
    parser.add_argument("--adv_warmup_epochs", type=int, default=10,
                        help="Linear warmup epochs for GRL lambda (0→target). 0 = start at target immediately.")
    parser.add_argument("--adv_val_weight", type=float, default=0.0,
                        help="Subject CE weight in VAL loss for checkpoint selection. "
                             "0 = phon-only checkpoint (recommended for LOSO mean maximization).")

    # phase 1
    parser.add_argument("--epochs_phase1", type=int, default=80)
    parser.add_argument("--lr_phase1", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--patience_phase1", type=int, default=8,
                        help="Early stopping patience for phase1. 0 disables.")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_channels", type=int, default=10)  # auto set per split
    parser.add_argument("--repro_paper10", action="store_true",
                        help="Force channel_mode=paper10, n_channels=10.")
    parser.add_argument("--augment", action="store_true",
                        help="Enable training data augmentation.")
    parser.add_argument("--augment_start", type=int, default=0,
                        help="Epoch from which augmentation is enabled (0=from the start).")

    # Phase1 imbalance handling
    parser.add_argument("--use_task_pos_weight", action="store_true",
                        help="Task-wise pos_weight in BCEWithLogitsLoss (helps vowel/bilabial).")
    parser.add_argument("--pos_weight_clip", type=float, default=10.0,
                        help="Clip pos_weight to [1, clip]. 0 disables.")
    parser.add_argument("--pos_weight_sqrt", action="store_true",
                        help="Use sqrt(pos_weight) to reduce extreme weights.")

    # phase 2
    parser.add_argument("--epochs_phase2", type=int, default=200)
    parser.add_argument("--lr_phase2", type=float, default=1e-3)
    parser.add_argument("--patience_phase2", type=int, default=10)
    parser.add_argument("--dae_noise_std", type=float, default=0.02)

    # xgb (paper settings)
    parser.add_argument("--xgb_estimators", type=int, default=5000)
    parser.add_argument("--xgb_depth", type=int, default=10)
    parser.add_argument("--xgb_lr", type=float, default=0.1)
    parser.add_argument("--xgb_lambda", type=float, default=0.3)
    parser.add_argument("--xgb_n_jobs", type=int, default=-1)

    # Phase3 imbalance + threshold tuning
    parser.add_argument("--xgb_use_scale_pos_weight", action="store_true",
                        help="Per-task scale_pos_weight=neg/pos in XGB (helps imbalanced tasks).")
    parser.add_argument("--thr_min",  type=float, default=0.05)
    parser.add_argument("--thr_max",  type=float, default=0.95)
    parser.add_argument("--thr_steps",type=int,   default=19)
    parser.add_argument("--print_thresholds", action="store_true")

    # loader perf
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.repro_paper10:
        if args.channel_mode != "paper10":
            print("[Guard] --repro_paper10 -> forcing channel_mode=paper10")
        args.channel_mode = "paper10"
        args.n_channels = 10

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtcwt_cfg = DTCWTConfig()

    def infer_n_channels(ds):
        sample = ds[0]
        return int(sample["eeg"].shape[1])

    if args.split_mode == "precomputed":
        tr_ds = PrecomputedDataset(f"{args.data_dir}/train.pt", normalize=False)
        va_ds = PrecomputedDataset(f"{args.data_dir}/val.pt", normalize=False)
        te_ds = PrecomputedDataset(f"{args.data_dir}/test.pt", normalize=False)
        args.n_channels = infer_n_channels(tr_ds)
        acc, task_acc = run_pipeline(tr_ds, va_ds, te_ds, device, args)
        print(f"Final test accuracy: {acc:.4f}")
        return

    if args.split_mode == "loso_cv":
        results = []
        for tr_ds, va_ds, te_ds, _, test_sid in loso_splits(
            args.raw_data_dir,
            seed=args.seed,
            verbose=False,
            augment_train=args.augment,
            dtcwt_cfg=dtcwt_cfg,
            channel_mode=args.channel_mode,
            feature_mode=args.feature_mode,
            ccv_repr=args.ccv_repr,
            ccv_reject_alpha=args.ccv_reject_alpha,
            cov_bounded_scale=args.cov_bounded_scale,
            apply_ica=args.apply_ica,
            ica_method=args.ica_method,
            ica_n_components=None if args.ica_n_components == 0 else int(args.ica_n_components),
            ica_random_state=args.ica_random_state,
            ica_max_iter=args.ica_max_iter,
        ):
            if args.loso_sid is not None and int(test_sid) != int(args.loso_sid):
                continue

            args.n_channels = infer_n_channels(tr_ds)
            # Stash fold id so train_phase1/2/3 can write fold-specific checkpoints.
            args.current_test_sid = int(test_sid)
            print(f"\n=== Evaluating LOSO fold | Test SID: {test_sid} | C={args.n_channels} ===")
            acc, acc_per_task = run_pipeline(tr_ds, va_ds, te_ds, device, args)
            results.append(acc)

            print("--- XGBoost Test Results ---")
            for name, a in zip(["vowel", "nasal", "bilabial", "iy", "uw"], acc_per_task):
                print(f"  {name:<8}: {float(a):.4f}")
            print(f"  Mean accuracy: {acc:.4f}")

        if len(results) > 0:
            print("\n--- LOSO CV Results ---")
            print(f"Mean Accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}")
        return

    # subject or trial
    split_func = create_trial_random_splits if args.split_mode == "trial" else create_random_splits
    tr_ds, va_ds, te_ds, _ = split_func(
        args.raw_data_dir,
        seed=args.seed,
        verbose=True,
        augment_train=args.augment,
        dtcwt_cfg=dtcwt_cfg,
        channel_mode=args.channel_mode,
        feature_mode=args.feature_mode,
        ccv_repr=args.ccv_repr,
        ccv_reject_alpha=args.ccv_reject_alpha,
        cov_bounded_scale=args.cov_bounded_scale,
        apply_ica=args.apply_ica,
        ica_method=args.ica_method,
        ica_n_components=None if args.ica_n_components == 0 else int(args.ica_n_components),
        ica_random_state=args.ica_random_state,
        ica_max_iter=args.ica_max_iter,
    )
    args.n_channels = infer_n_channels(tr_ds)
    acc, acc_per_task = run_pipeline(tr_ds, va_ds, te_ds, device, args)

    print("--- XGBoost Test Results ---")
    for name, a in zip(["vowel", "nasal", "bilabial", "iy", "uw"], acc_per_task):
        print(f"  {name:<8}: {float(a):.4f}")
    print(f"  Mean accuracy: {acc:.4f}")
    print(f"Final test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
