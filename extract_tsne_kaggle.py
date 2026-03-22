# extract_tsne_kaggle.py  (v2 — P1+P2 DAE 정확한 latent 추출)
#
# 필요 파일:
#   - phase1_best_sid{N}.pt  (train_p1p2_checkpoints.py 로 생성)
#   - phase2_best_sid{N}.pt  (같은 스크립트로 생성)
#
# 파이프라인 정확한 구조:
#   PCA  : random CNN+LSTM → 1152d → PCA(32)
#   DAE  : trained CNN+LSTM (P1) → 1152d → DAE encoder (P2) → 32d latent
#
# 캐글 셀:
#   exec(open("extract_tsne_kaggle.py").read())
#
# 출력:
#   /kaggle/working/tsne_pca_dae.png        (dark, 포스터용)
#   /kaggle/working/tsne_pca_dae_white.png  (white, 논문용)
#   /kaggle/working/tsne_features.npz

import os, sys, warnings
warnings.filterwarnings('ignore')

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle/input")
for _p in ["/kaggle/input/datasets/hdwmmmlc/codeupload", "/kaggle/working", os.getcwd()]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from kara_one_dataset import loso_splits
from model_manifold import SpatialCNN, TemporalLSTM, EEGPhonologicalManifoldNet
from train_stage2_manifold import DTCWTConfig, _KAGGLE_DATA_DIR

# ── 경로 / 설정 ────────────────────────────────────────────────────────────────
DATA_DIR     = _KAGGLE_DATA_DIR if IS_KAGGLE else "/data/kara_one"
CKPT_DIR     = "/kaggle/working/checkpoints_p1p2"   # train_p1p2_checkpoints.py 와 동일
OUT_DIR      = "/kaggle/working"

CHANNEL_MODE    = "all"
SEED            = 42
LATENT_DIM      = 32
BATCH_SIZE      = 64
N_WORKERS       = 2
TSNE_PERPLEXITY = 30
TSNE_ITER       = 1000

TASK_NAMES   = ['vowel', 'nasal', 'bilabial', '/iy/', '/uw/']
TASK_COLORS  = ['#3A7DD8', '#0A9070', '#E08020', '#C030A0', '#D04028']
SUBJ_MARKERS = ['o', 's', '^', 'D']
SUBJ_GROUPS  = [list(range(0,4)), list(range(4,8)), list(range(8,11)), list(range(11,14))]
SUBJ_LABELS  = ['SID 0–3', 'SID 4–7', 'SID 8–10', 'SID 11–13']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


# ── DAE 정의 (train_stage2_manifold.DeepAutoencoder 와 동일) ──────────────────
class _DAE(nn.Module):
    def __init__(self, input_dim=1152, latent_dim=32):
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


# ── feature 추출 ──────────────────────────────────────────────────────────────

def extract_pca_features(n_ch, loader):
    """
    Pipeline A: random-init CNN+LSTM → 1152d (고정, 학습 없음)
    seed 고정으로 완전 결정론적
    """
    torch.manual_seed(SEED)
    cnn  = SpatialCNN(n_channels=n_ch).to(device).eval()
    lstm = TemporalLSTM(in_dim=n_ch).to(device).eval()
    feats, labels, sids = [], [], []
    with torch.no_grad():
        for batch in loader:
            eeg  = batch["eeg"].to(device)
            corr = batch["corr"].to(device)
            mask = batch.get("corr_mask", torch.ones_like(batch["corr"])).to(device)
            attn = batch["attention_mask"].to(device)
            merged = torch.cat([cnn(corr, mask), lstm(eeg, attn)], dim=1)
            feats.append(merged.cpu().numpy())
            labels.append(batch["phon"].cpu().numpy())
            sids.append(batch["subject_id"].cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels), np.concatenate(sids)


def extract_dae_latent(p1_model, dae_model, loader):
    """
    Pipeline B: trained P1 CNN+LSTM → 1152d → trained DAE encoder → 32d latent
    """
    p1_model.eval(); dae_model.eval()
    latents, labels, sids = [], [], []
    with torch.no_grad():
        for batch in loader:
            eeg  = batch["eeg"].to(device)
            corr = batch["corr"].to(device)
            mask = batch.get("corr_mask", torch.ones_like(batch["corr"])).to(device)
            attn = batch["attention_mask"].to(device)
            # Step 1: P1 → 1152d merged
            _, merged, _ = p1_model.encode(eeg, corr, mask, attn)
            # Step 2: DAE encoder → 32d latent
            _, z = dae_model(merged)
            latents.append(z.cpu().numpy())
            labels.append(batch["phon"].cpu().numpy())
            sids.append(batch["subject_id"].cpu().numpy())
    return np.concatenate(latents), np.concatenate(labels), np.concatenate(sids)


# ── 전체 데이터 수집 (LOSO held-out) ──────────────────────────────────────────
print("="*60)
print("Collecting held-out features from all LOSO folds...")
print("="*60)

dtcwt_cfg = DTCWTConfig()

all_feats_pca, all_latents_dae, all_labels, all_sids = [], [], [], []
n_ch = None
dae_missing = []

for tr_ds, va_ds, te_ds, _, test_sid in loso_splits(
    DATA_DIR, seed=SEED, verbose=False, dtcwt_cfg=dtcwt_cfg,
    channel_mode=CHANNEL_MODE, feature_mode="raw", ccv_repr="cov_bounded",
    ccv_reject_alpha=0.10, cov_bounded_scale=0.10, apply_ica=False,
):
    if n_ch is None:
        n_ch = int(tr_ds[0]["eeg"].shape[1])

    te_ldr = DataLoader(
        te_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=te_ds.collate_fn, num_workers=N_WORKERS,
    )

    print(f"\n── SID {test_sid}  (C={n_ch}, n_test={len(te_ds)}) ──")

    # PCA features (항상 가능)
    f, l, s = extract_pca_features(n_ch, te_ldr)
    all_feats_pca.append(f)
    all_labels.append(l)
    all_sids.append(s)
    print(f"  PCA feat : {f.shape}")

    # DAE latent (체크포인트 있을 때만)
    p1_path  = os.path.join(CKPT_DIR, f"phase1_best_sid{int(test_sid)}.pt")
    p2_path  = os.path.join(CKPT_DIR, f"phase2_best_sid{int(test_sid)}.pt")

    if os.path.isfile(p1_path) and os.path.isfile(p2_path):
        # P1 모델 로드
        p1 = EEGPhonologicalManifoldNet(
            n_channels=n_ch, n_subjects=13, n_tasks=5,
            latent_dim=LATENT_DIM, grl_lambda=0.0,
        ).to(device)
        p1.load_state_dict(torch.load(p1_path, map_location=device))

        # P2 (DAE) 로드
        dae = _DAE(input_dim=1152, latent_dim=LATENT_DIM).to(device)
        dae.load_state_dict(torch.load(p2_path, map_location=device))

        z, _, _ = extract_dae_latent(p1, dae, te_ldr)
        all_latents_dae.append(z)
        print(f"  DAE latent: {z.shape}  ✓")
    else:
        missing = []
        if not os.path.isfile(p1_path): missing.append("phase1")
        if not os.path.isfile(p2_path): missing.append("phase2")
        print(f"  DAE latent: SKIP (missing: {', '.join(missing)})")
        all_latents_dae.append(None)
        dae_missing.append(int(test_sid))

# ── 통합 ───────────────────────────────────────────────────────────────────────
X_merged = np.concatenate(all_feats_pca, axis=0)
y_all    = np.concatenate(all_labels, axis=0)
s_all    = np.concatenate(all_sids, axis=0).astype(int).ravel()

print(f"\nTotal samples: {len(s_all)}, subjects: {len(np.unique(s_all))}")

# PCA(32) on all merged features
pca = PCA(n_components=LATENT_DIM, random_state=SEED)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X_merged))
print(f"PCA(32) explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# DAE latent
dae_available = all(x is not None for x in all_latents_dae)
if dae_available:
    X_dae = StandardScaler().fit_transform(np.concatenate(all_latents_dae, axis=0))
    print(f"DAE latent: all {len(all_latents_dae)} folds loaded ✓")
elif len(dae_missing) < 14:
    # 일부만 있으면 해당 subject만으로 partial plot
    valid_idx = [i for i, x in enumerate(all_latents_dae) if x is not None]
    X_dae = StandardScaler().fit_transform(
        np.concatenate([all_latents_dae[i] for i in valid_idx], axis=0)
    )
    # subject / label도 해당 SID만
    valid_sids = [i for i in range(14) if i not in dae_missing]
    mask_valid = np.isin(s_all, valid_sids)
    s_dae = s_all[mask_valid]
    y_dae = y_all[mask_valid]
    print(f"DAE latent: partial ({len(valid_sids)}/14 subjects)")
else:
    X_dae = None
    s_dae = s_all
    y_dae = y_all
    print("DAE latent: no checkpoints found — DAE panel skipped")
    print(f"→ 먼저 train_p1p2_checkpoints.py 실행 후 다시 시도하세요")

# 공통 dominant task
priority = [4, 3, 2, 1, 0]
def dominant_task(t):
    for p in priority:
        if t[p] == 1: return p
    return 0

dom_task_pca = np.array([dominant_task(t) for t in y_all])
if X_dae is not None:
    dom_task_dae = np.array([dominant_task(t) for t in (y_dae if not dae_available else y_all)])
    s_dae_plot   = s_dae if not dae_available else s_all

# ── 저장 ───────────────────────────────────────────────────────────────────────
np.savez(
    os.path.join(OUT_DIR, "tsne_features.npz"),
    X_pca=X_pca,
    X_dae=X_dae if X_dae is not None else np.array([]),
    subjects=s_all,
    dom_task=dom_task_pca,
    task_labels=y_all,
)
print(f"Features saved → {OUT_DIR}/tsne_features.npz")

# ── t-SNE ──────────────────────────────────────────────────────────────────────
def run_tsne(X, label):
    print(f"Running t-SNE ({label})...")
    return TSNE(
        n_components=2, perplexity=TSNE_PERPLEXITY, max_iter=TSNE_ITER,
        random_state=SEED, learning_rate='auto', init='pca',
    ).fit_transform(X)

Z_pca = run_tsne(X_pca, "PCA features")
Z_dae = run_tsne(X_dae, "DAE latent") if X_dae is not None else None

# ── Plot ───────────────────────────────────────────────────────────────────────
def scatter_panel(ax, Z, s_arr, dom_arr, bg='dark'):
    fc   = '#1A1D2C' if bg == 'dark' else '#F6F8FC'
    tc   = 'white'   if bg == 'dark' else '#0C0F18'
    subc = '#6B7494'
    tickc= '#555D72' if bg == 'dark' else '#6B7494'
    ec   = '#252836' if bg == 'dark' else '#DDE3EE'

    ax.set_facecolor(fc)
    ax.tick_params(colors=tickc, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(ec)
    ax.set_xlabel('t-SNE 1', color=tickc, fontsize=8)
    ax.set_ylabel('t-SNE 2', color=tickc, fontsize=8)

    for gi, grp in enumerate(SUBJ_GROUPS):
        m = SUBJ_MARKERS[gi]
        gm = np.isin(s_arr, grp)
        for ti in range(5):
            mask = gm & (dom_arr == ti)
            if mask.sum() == 0: continue
            ax.scatter(Z[mask,0], Z[mask,1],
                       c=TASK_COLORS[ti], marker=m,
                       s=26, alpha=0.70, linewidths=0)


def add_legends(fig, bg='dark'):
    lc = '#C2C0B6' if bg == 'dark' else '#3E4460'
    fc = '#1A1D2C' if bg == 'dark' else '#F6F8FC'
    ec = '#252836' if bg == 'dark' else '#DDE3EE'

    tp = [mpatches.Patch(color=TASK_COLORS[i], label=TASK_NAMES[i]) for i in range(5)]
    sh = [plt.scatter([], [], marker=SUBJ_MARKERS[i], color='#9AA3B8',
                      s=32, label=SUBJ_LABELS[i]) for i in range(4)]

    l1 = fig.legend(handles=tp, title='Task (dominant, color)',
                    loc='lower left', bbox_to_anchor=(0.06, -0.05),
                    ncol=5, fontsize=9, title_fontsize=8,
                    labelcolor=lc, facecolor=fc, edgecolor=ec, framealpha=0.9)
    l1.get_title().set_color('#9AA3B8')
    l2 = fig.legend(handles=sh, title='Subject group (shape)',
                    loc='lower right', bbox_to_anchor=(0.94, -0.05),
                    ncol=4, fontsize=9, title_fontsize=8,
                    labelcolor=lc, facecolor=fc, edgecolor=ec, framealpha=0.9)
    l2.get_title().set_color('#9AA3B8')
    fig.add_artist(l1)


n_panels = 2 if Z_dae is not None else 1

for bg in ['dark', 'white']:
    bgc = '#0F1117' if bg == 'dark' else 'white'
    fig, axes = plt.subplots(1, n_panels, figsize=(7*n_panels, 6.2))
    if n_panels == 1: axes = [axes]
    fig.patch.set_facecolor(bgc)
    tc = 'white' if bg == 'dark' else '#0C0F18'

    # PCA panel
    scatter_panel(axes[0], Z_pca, s_all, dom_task_pca, bg=bg)
    axes[0].set_title('Fig A — PCA(32) features', color=tc,
                      fontsize=12, fontweight='bold', pad=8)
    axes[0].text(0.5, 1.01,
        f'random-init CNN+LSTM → 1152d → PCA(32)  (expvar={pca.explained_variance_ratio_.sum():.3f})',
        transform=axes[0].transAxes, ha='center', va='bottom',
        color='#6B7494', fontsize=8, style='italic')

    # DAE panel
    if Z_dae is not None:
        scatter_panel(axes[1], Z_dae, s_dae_plot, dom_task_dae, bg=bg)
        axes[1].set_title('Fig B — DAE latent (32d)', color=tc,
                           fontsize=12, fontweight='bold', pad=8)
        axes[1].text(0.5, 1.01,
            'trained CNN+LSTM (P1) → 1152d → DAE encoder (P2) → 32d latent',
            transform=axes[1].transAxes, ha='center', va='bottom',
            color='#6B7494', fontsize=8, style='italic')

    add_legends(fig, bg=bg)

    n_subj_dae = len(np.unique(s_dae_plot)) if Z_dae is not None else 0
    fig.text(0.5, -0.07,
        f't-SNE of actual Kara-One LOSO held-out features  ·  '
        f'N={len(s_all)} trials  ·  14 subjects  ·  '
        f'perplexity={TSNE_PERPLEXITY}  ·  seed={SEED}',
        ha='center', color='#6B7494', fontsize=8.5, style='italic')

    suffix = '' if bg == 'dark' else '_white'
    out_path = os.path.join(OUT_DIR, f"tsne_pca_dae{suffix}.png")
    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=bgc, edgecolor='none')
    plt.close()
    print(f"Saved: {out_path}")

print(f"\n{'='*50}")
print(f"Done!")
if dae_missing:
    print(f"⚠ DAE missing for SIDs: {dae_missing}")
    print(f"  → run train_p1p2_checkpoints.py 먼저")