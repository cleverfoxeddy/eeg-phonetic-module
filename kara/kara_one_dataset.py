# kara_one_dataset.py
from __future__ import annotations

import glob
import os
import random
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import torch
from pymatreader import read_mat
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=RuntimeWarning)


WORD_CLASSES: Dict[str, int] = {
    "piy": 0, "tiy": 1, "diy": 2, "gnaw": 3, "knew": 4,
    "pat": 5, "pot": 6, "iy": 7, "uw": 8, "m": 9, "n": 10,
}
FEATURE_NAMES: List[str] = ["vowel", "nasal", "bilabial", "iy", "uw"]
NUM_TASKS: int = 5

PHONOLOGICAL_FEATURES: Dict[str, List[int]] = {
    "piy":  [1, 0, 1, 1, 0], "tiy":  [1, 0, 0, 1, 0], "diy":  [1, 0, 0, 1, 0],
    "gnaw": [0, 1, 0, 0, 0], "knew": [1, 1, 0, 0, 1], "pat":  [0, 0, 1, 0, 0],
    "pot":  [0, 0, 1, 0, 0], "iy":   [1, 0, 0, 1, 0], "uw":   [1, 0, 0, 0, 1],
    "m":    [0, 1, 1, 0, 0], "n":    [0, 1, 0, 0, 0],
}

_TARGET_SFREQ: int = 1000
_MAX_CHANNELS: int = 64

# paper 10 channels (legacy mode)
PAPER_CHANNELS: List[str] = ["T7", "C5", "C3", "CP5", "CP3", "CP1", "P3", "FT8", "FC6", "C4"]
N_CHANNELS: int = len(PAPER_CHANNELS)


@dataclass
class DTCWTConfig:
    nlevels: int = 6
    strict_qshift_len10: bool = False
    qshift_candidates: Tuple[str, ...] = ("qshift_a",)
    biort_candidates: Tuple[str, ...] = ("near_sym_a", "near_sym_b", "near_sym_c", "antonini", "legall")


def _as_complex(detail: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(detail):
        return detail
    d = np.asarray(detail)
    if d.ndim >= 2 and d.shape[-1] == 2:
        return d[..., 0] + 1j * d[..., 1]
    raise ValueError("Unsupported format")


def make_dtcwt_transform(cfg: DTCWTConfig, console: Optional[Console] = None):
    import dtcwt
    for qname in cfg.qshift_candidates:
        for bname in cfg.biort_candidates:
            try:
                return dtcwt.Transform1d(biort=bname, qshift=qname)
            except Exception:
                continue
    raise RuntimeError("Cannot create DTCWT transform.")


def extract_beta_dtcwt(eeg_np: np.ndarray, transform, nlevels: int = 6) -> np.ndarray:
    T, C = eeg_np.shape
    L = 2 ** nlevels
    T2 = (T // L) * L
    if T2 != T:
        eeg_np = eeg_np[:T2, :]
    feats = []
    for ch in range(C):
        x = eeg_np[:, ch].astype(np.float32)
        res = transform.forward(x, nlevels=nlevels)
        detail = res.highpasses[-1]
        z = _as_complex(detail)
        mag = np.abs(z).astype(np.float32)
        feats.append(mag)
    return np.column_stack(feats)


# ---------------- CCV helpers ----------------
def compute_ccv_full(eeg: torch.Tensor):
    """
    eeg: [T, C] in uV
    returns: corr [C,C] (-1..1), cov [C,C], std [C]
    """
    if eeg.dim() == 3:
        eeg = eeg[0]
    x = eeg - eeg.mean(dim=0, keepdim=True)
    cov = torch.mm(x.t(), x) / max(1, x.size(0) - 1)
    std = torch.sqrt(torch.diag(cov)).clamp(min=1e-8)
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
    corr = corr.clamp(-1.0, 1.0)
    return corr, cov, std


def cov_to_bounded_ccv(cov: torch.Tensor, scale: float = 0.10, eps: float = 1e-8) -> torch.Tensor:
    """
    Stable covariance representation:
      signed log1p(|cov|) then tanh(scale * x) -> [-1,1]
    """
    x = torch.sign(cov) * torch.log1p(cov.abs().clamp(min=eps))
    return torch.tanh(scale * x).clamp(-1.0, 1.0)


def ccv_channel_reject_mask(cov: torch.Tensor, alpha: float = 0.10) -> torch.Tensor:
    """
    Paper-style heuristic:
      keep channel i if mean(|cov(i,j)|, j!=i) >= alpha * |cov(i,i)|
    """
    C = cov.size(0)
    auto = torch.diag(cov).abs().clamp(min=1e-8)
    cross = cov.abs().clone()
    cross.fill_diagonal_(0.0)
    cross_mean = cross.sum(dim=1) / max(C - 1, 1)
    return cross_mean >= (alpha * auto)


def get_channel_indices(raw_info: dict) -> List[int]:
    ch_names = raw_info["ch_names"]
    indices, missing = [], []
    for name in PAPER_CHANNELS:
        if name in ch_names:
            indices.append(ch_names.index(name))
        else:
            found = [i for i, n in enumerate(ch_names) if name in n]
            if found:
                indices.append(found[0])
            else:
                missing.append(name)
    if missing:
        raise ValueError(f"Channels not found: {missing}")
    return indices


# ---------------- Loader ----------------
class KaraOneDataLoader:
    def __init__(
        self,
        raw_data_dir: str,
        subjects: Any = "all",
        epoch_type: str = "thinking",
        verbose: bool = False,
        random_state: int = 42,
        dtcwt_cfg: Optional[DTCWTConfig] = None,
        channel_mode: str = "paper10",         # "paper10" or "all"
        ensure_multiple_of_4: bool = True,     # for SpatialCNN pooling stability
        apply_ica: bool = True,                # ocular removal (best-effort)
        ica_method: str = "fastica",
        ica_n_components: Optional[int] = None,
        ica_random_state: int = 97,
        ica_max_iter: int = 512,
    ):
        self.raw_data_dir = raw_data_dir
        self.epoch_type = epoch_type
        self.verbose = verbose
        self.console = Console()
        self.random_state = int(random_state)

        self.all_subjects = self._discover_subjects()
        self.subjects = self._resolve_subjects(subjects)
        self._subject_map = {name: idx for idx, name in enumerate(self.subjects)}
        self.subject_name_map = {idx: name for name, idx in self._subject_map.items()}

        self._data: List[Dict[str, Any]] = []
        self.channel_indices: Optional[List[int]] = None
        self.channel_names: Optional[List[str]] = None

        self.dtcwt_cfg = dtcwt_cfg or DTCWTConfig()
        self.channel_mode = channel_mode
        self.ensure_multiple_of_4 = bool(ensure_multiple_of_4)

        self.apply_ica = bool(apply_ica)
        self.ica_method = str(ica_method)
        self.ica_n_components = ica_n_components
        self.ica_random_state = int(ica_random_state)
        self.ica_max_iter = int(ica_max_iter)

        self._dtcwt_transform = make_dtcwt_transform(self.dtcwt_cfg, console=self.console)

    def _discover_subjects(self) -> List[str]:
        found = []
        for pat in ("MM*", "P*"):
            found += [
                os.path.basename(p)
                for p in glob.glob(os.path.join(self.raw_data_dir, "**", pat), recursive=True)
                if os.path.isdir(p)
            ]
        return sorted(set(found))

    def _resolve_subjects(self, subjects: Any) -> List[str]:
        if subjects == "all":
            return list(self.all_subjects)
        if isinstance(subjects, list):
            return list(subjects)
        raise ValueError("Invalid subjects format")

    def load_data(self, max_per_subject: Optional[int] = None) -> List[Dict[str, Any]]:
        for name in self.subjects:
            sid = self._subject_map[name]
            try:
                self._data.extend(self._load_subject(name, sid, max_per_subject))
            except Exception as e:
                self.console.print(f"[red]SKIP {name}: {e}[/]")
        return self._data

    def _maybe_apply_ica(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Best-effort ocular artifact removal.
        - If EOG channels exist: use find_bads_eog
        - Else: attempt heuristic on frontal channels; if fails, skip silently.
        """
        if not self.apply_ica:
            return raw

        try:
            from mne.preprocessing import ICA
        except Exception:
            if self.verbose:
                self.console.print("[yellow]mne.preprocessing.ICA not available; skip ICA.[/]")
            return raw

        try:
            # ICA requires reasonable high-pass; we keep as-is and let mne handle
            ica = ICA(
                n_components=self.ica_n_components,
                method=self.ica_method,
                random_state=self.ica_random_state,
                max_iter=self.ica_max_iter,
            )
            ica.fit(raw)

            picks_eog = mne.pick_types(raw.info, eog=True)
            if len(picks_eog) > 0:
                eog_inds, _ = ica.find_bads_eog(raw)
                ica.exclude = list(set(eog_inds))
            else:
                # Heuristic: try frontal-like channels if present
                frontal = [ch for ch in raw.ch_names if any(k in ch.upper() for k in ["FP1", "FP2", "AF", "FZ", "F1", "F2"])]
                if len(frontal) >= 1:
                    # Use correlation with the mean frontal signal as pseudo-EOG
                    pseudo = raw.copy().pick_channels(frontal).get_data().mean(axis=0, keepdims=True)
                    pseudo_raw = mne.io.RawArray(pseudo, mne.create_info(["PSEUDO_EOG"], raw.info["sfreq"], ch_types=["eog"]))
                    eog_inds, _ = ica.find_bads_eog(pseudo_raw)
                    ica.exclude = list(set(eog_inds))
                else:
                    # No safe heuristic
                    ica.exclude = []

            if len(getattr(ica, "exclude", [])) > 0:
                raw = ica.apply(raw.copy())
                if self.verbose:
                    self.console.print(f"[green]ICA applied. excluded comps: {ica.exclude}[/]")
            return raw
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]ICA failed; skip. ({e})[/]")
            return raw

    def _load_subject(self, name: str, sid: int, max_n: Optional[int]) -> List[Dict[str, Any]]:
        dirs = glob.glob(os.path.join(self.raw_data_dir, "**", name), recursive=True)
        if not dirs:
            raise RuntimeError("Subject folder not found")
        d = dirs[0]
        cnt_files = glob.glob(os.path.join(d, "**", "*.cnt"), recursive=True)
        inds_files = glob.glob(os.path.join(d, "**", "epoch_inds.mat"), recursive=True)
        feat_files = glob.glob(os.path.join(d, "**", "all_features_simple.mat"), recursive=True)
        if not cnt_files or not inds_files or not feat_files:
            raise RuntimeError("Missing cnt/epoch_inds/all_features_simple")

        raw = mne.io.read_raw_cnt(cnt_files[0], preload=True, verbose="ERROR")

        if len(raw.ch_names) > _MAX_CHANNELS:
            raw.pick(raw.ch_names[:_MAX_CHANNELS])

        if self.channel_mode == "all":
            try:
                raw.pick_types(eeg=True)
            except Exception:
                pass

        # ICA/BSS ocular removal (best-effort)
        raw = self._maybe_apply_ica(raw)

        # choose channels
        if self.channel_indices is None:
            if self.channel_mode == "all":
                self.channel_indices = list(range(len(raw.ch_names)))
                self.channel_names = [raw.ch_names[i] for i in self.channel_indices]
            else:
                # paper10 mode: strict — no silent fallback
                self.channel_indices = get_channel_indices(raw.info)
                self.channel_names = [raw.ch_names[i] for i in self.channel_indices]

            # enforce multiple-of-4 for SpatialCNN pooling
            # SKIP trimming for paper10: paper specifies exactly 10 channels (not a multiple of 4)
            # and trimming to 8 breaks reproduction.
            if self.ensure_multiple_of_4 and self.channel_mode != "paper10" and len(self.channel_indices) >= 4:
                r = len(self.channel_indices) % 4
                if r != 0:
                    new_len = len(self.channel_indices) - r
                    self.channel_indices = self.channel_indices[:new_len]
                    self.channel_names = self.channel_names[:new_len]
        else:
            if len(raw.ch_names) < max(self.channel_indices) + 1:
                raise RuntimeError("Channel length mismatch across subjects")

        # preprocessing: bandpass + resample
        orig_sf = float(raw.info["sfreq"])
        nyquist = orig_sf / 2.0
        hp = min(50.0, nyquist - 1.0)
        raw.filter(1.0, hp, verbose="ERROR")
        if orig_sf != float(_TARGET_SFREQ):
            raw.resample(_TARGET_SFREQ, npad="auto", verbose="ERROR")

        ratio = float(_TARGET_SFREQ) / orig_sf

        mat_i = read_mat(inds_files[0])
        mat_f = read_mat(feat_files[0])
        idx_list = mat_i.get(f"{self.epoch_type}_inds", [])
        prompts = mat_f.get("all_features", {}).get("prompts", [])

        records = []
        n = min(len(idx_list), max_n) if max_n else len(idx_list)
        for i in range(n):
            rec = self._parse_trial(raw, idx_list[i], prompts, i, sid, ratio, raw.n_times)
            if rec is not None:
                records.append(rec)
        return records

    def _parse_trial(self, raw, trial_inds, prompts, i, sid, ratio, total_samples):
        try:
            trial_inds = np.asarray(trial_inds, dtype=float)
            start = int((np.min(trial_inds) - 1) * ratio)
            end = int(np.max(trial_inds) * ratio)
            start = max(0, start)
            end = min(total_samples, end)
            if end - start < 128:
                return None

            eeg_all = raw.get_data(start=start, stop=end)  # [C, T] in Volts
            eeg_sel = eeg_all[self.channel_indices, :].T.astype(np.float32)  # [T, C]
            if not np.isfinite(eeg_sel).all() or np.all(eeg_sel == 0):
                return None

            prompt_str = str(prompts[i]) if (prompts is not None and i < len(prompts)) else ""
            word = re.sub(r"[^a-zA-Z]", "", prompt_str).lower()
            if word not in WORD_CLASSES:
                return None

            rec = {"eeg": eeg_sel, "token_label": WORD_CLASSES[word], "subject_id": sid, "word": word}
            for fn, fv in zip(FEATURE_NAMES, PHONOLOGICAL_FEATURES[word]):
                rec[fn] = fv
            return rec
        except Exception:
            return None


# ---------------- Dataset ----------------
class KaraOnePhoneticDataset(Dataset):
    FEATURE_NAMES = FEATURE_NAMES
    NUM_TASKS = NUM_TASKS

    def __init__(
        self,
        records: List[Dict[str, Any]],
        channel_indices: List[int],
        dtcwt_transform,
        dtcwt_cfg: DTCWTConfig,
        indices: Optional[List[int]] = None,
        augment: bool = False,
        feature_mode: str = "raw",          # "dtcwt_beta" or "raw"
        ccv_repr: str = "cov_bounded",      # "corr" or "cov_bounded"
        ccv_reject_alpha: float = 0.10,
        cov_bounded_scale: float = 0.10,
    ):
        self._records = records
        self._indices = indices if indices is not None else list(range(len(records)))
        self.augment = augment

        self.channel_indices = channel_indices
        self._dtcwt_tfm = dtcwt_transform
        self._dtcwt_cfg = dtcwt_cfg

        self.feature_mode = feature_mode
        self.ccv_repr = ccv_repr
        self.ccv_reject_alpha = float(ccv_reject_alpha)
        self.cov_bounded_scale = float(cov_bounded_scale)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self._records[self._indices[idx]]
        eeg_vol = r["eeg"]  # [T,C] Volts

        # feature extraction (time series)
        if self.feature_mode == "dtcwt_beta":
            try:
                beta_np = extract_beta_dtcwt(eeg_vol, transform=self._dtcwt_tfm, nlevels=self._dtcwt_cfg.nlevels)
            except ValueError:
                L = 2 ** self._dtcwt_cfg.nlevels
                padded = np.zeros((L, eeg_vol.shape[1]), dtype=np.float32)
                T = min(eeg_vol.shape[0], L)
                padded[:T] = eeg_vol[:T]
                beta_np = extract_beta_dtcwt(padded, transform=self._dtcwt_tfm, nlevels=self._dtcwt_cfg.nlevels)
            eeg_uv = torch.from_numpy(beta_np).float() * 1e6  # uV
        elif self.feature_mode == "raw":
            eeg_uv = torch.from_numpy(eeg_vol).float() * 1e6  # uV
        else:
            raise ValueError(f"Unknown feature_mode={self.feature_mode}")

        # CCV
        corr, cov, _ = compute_ccv_full(eeg_uv)
        ch_keep = ccv_channel_reject_mask(cov, alpha=self.ccv_reject_alpha)
        C = corr.size(0)
        corr_mask = (ch_keep.view(C, 1) & ch_keep.view(1, C)).float()

        if self.ccv_repr == "corr":
            ccv_in = corr
        elif self.ccv_repr == "cov_bounded":
            ccv_in = cov_to_bounded_ccv(cov, scale=self.cov_bounded_scale)
        else:
            raise ValueError(f"Unknown ccv_repr={self.ccv_repr}")

        ccv_in = (ccv_in * corr_mask).clamp(-1.0, 1.0)

        phon = torch.tensor([r[f] for f in FEATURE_NAMES], dtype=torch.float32)

        return {
            "eeg": eeg_uv,  # [T,C]
            "corr": ccv_in,  # [C,C]
            "corr_mask": corr_mask,  # [C,C]
            "ch_keep": ch_keep,  # [C]
            "phon": phon,  # [5]
            "token_label": torch.tensor(r["token_label"], dtype=torch.long),
            "subject_id": torch.tensor(r["subject_id"], dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        eeg_list = [b["eeg"] for b in batch]
        eeg_padded = pad_sequence(eeg_list, batch_first=True)  # [B,Tmax,C]
        attn_mask = torch.zeros(len(batch), eeg_padded.size(1), dtype=torch.long)
        for i, eeg in enumerate(eeg_list):
            attn_mask[i, : eeg.size(0)] = 1

        return {
            "eeg": eeg_padded,
            "attention_mask": attn_mask,
            "corr": torch.stack([b["corr"] for b in batch]),
            "corr_mask": torch.stack([b["corr_mask"] for b in batch]),
            "phon": torch.stack([b["phon"] for b in batch]),
            "token_label": torch.stack([b["token_label"] for b in batch]),
            "subject_id": torch.stack([b["subject_id"] for b in batch]),
        }


# ---------------- Split helpers ----------------
def create_random_splits(
    raw_data_dir: str,
    epoch_type: str = "thinking",
    seed: int = 42,
    verbose: bool = True,
    augment_train: bool = False,
    dtcwt_cfg: Optional[DTCWTConfig] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    loso_test_sid: Optional[int] = None,
    channel_mode: str = "all",
    feature_mode: str = "raw",
    ccv_repr: str = "cov_bounded",
    ccv_reject_alpha: float = 0.10,
    cov_bounded_scale: float = 0.10,
    apply_ica: bool = True,
    ica_method: str = "fastica",
    ica_n_components: Optional[int] = None,
    ica_random_state: int = 97,
    ica_max_iter: int = 512,
):
    if channel_mode not in ("paper10", "all"):
        raise ValueError(f"Invalid channel_mode={channel_mode!r}. Must be 'paper10' or 'all'.")
    loader = KaraOneDataLoader(
        raw_data_dir,
        epoch_type=epoch_type,
        verbose=verbose,
        random_state=seed,
        dtcwt_cfg=dtcwt_cfg,
        channel_mode=channel_mode,
        apply_ica=apply_ica,
        ica_method=ica_method,
        ica_n_components=ica_n_components,
        ica_random_state=ica_random_state,
        ica_max_iter=ica_max_iter,
    )
    records = loader.load_data()
    all_sids = sorted(set(int(r["subject_id"]) for r in records))

    if channel_mode == "paper10" and loader.channel_indices is not None:
        if len(loader.channel_indices) != 10:
            raise RuntimeError(
                f"[Guard] channel_mode=paper10 expects 10 channels, "
                f"but loader selected {len(loader.channel_indices)}. Check PAPER_CHANNELS mapping."
            )

    rng = np.random.RandomState(seed)
    if loso_test_sid is not None:
        remaining = [s for s in all_sids if s != loso_test_sid]
        rng.shuffle(remaining)
        n_rem = len(remaining)
        n_va = max(1, round(n_rem * (val_ratio / (train_ratio + val_ratio))))
        va_sids = set(remaining[:n_va])
        tr_sids = set(remaining[n_va:])
        te_sids = {loso_test_sid}
    else:
        sids_shuffled = list(all_sids)
        rng.shuffle(sids_shuffled)
        n = len(sids_shuffled)
        n_tr = max(1, round(n * train_ratio))
        n_va = max(1, round(n * val_ratio))
        n_te = max(1, n - n_tr - n_va)
        tr_sids = set(sids_shuffled[:n_tr])
        va_sids = set(sids_shuffled[n_tr:n_tr + n_va])
        te_sids = set(sids_shuffled[n_tr + n_va:n_tr + n_va + n_te])

    tr_idx = [i for i, r in enumerate(records) if int(r["subject_id"]) in tr_sids]
    va_idx = [i for i, r in enumerate(records) if int(r["subject_id"]) in va_sids]
    te_idx = [i for i, r in enumerate(records) if int(r["subject_id"]) in te_sids]

    cfg, tfm, ch = loader.dtcwt_cfg, loader._dtcwt_transform, loader.channel_indices
    return (
        KaraOnePhoneticDataset(records, ch, tfm, cfg, tr_idx, augment=augment_train,
                               feature_mode=feature_mode, ccv_repr=ccv_repr,
                               ccv_reject_alpha=ccv_reject_alpha, cov_bounded_scale=cov_bounded_scale),
        KaraOnePhoneticDataset(records, ch, tfm, cfg, va_idx, augment=False,
                               feature_mode=feature_mode, ccv_repr=ccv_repr,
                               ccv_reject_alpha=ccv_reject_alpha, cov_bounded_scale=cov_bounded_scale),
        KaraOnePhoneticDataset(records, ch, tfm, cfg, te_idx, augment=False,
                               feature_mode=feature_mode, ccv_repr=ccv_repr,
                               ccv_reject_alpha=ccv_reject_alpha, cov_bounded_scale=cov_bounded_scale),
        loader,
    )


def create_trial_random_splits(
    raw_data_dir: str,
    epoch_type: str = "thinking",
    seed: int = 42,
    verbose: bool = True,
    augment_train: bool = False,
    dtcwt_cfg: Optional[DTCWTConfig] = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    channel_mode: str = "all",
    feature_mode: str = "raw",
    ccv_repr: str = "cov_bounded",
    ccv_reject_alpha: float = 0.10,
    cov_bounded_scale: float = 0.10,
    apply_ica: bool = True,
    ica_method: str = "fastica",
    ica_n_components: Optional[int] = None,
    ica_random_state: int = 97,
    ica_max_iter: int = 512,
):
    if channel_mode not in ("paper10", "all"):
        raise ValueError(f"Invalid channel_mode={channel_mode!r}. Must be 'paper10' or 'all'.")
    loader = KaraOneDataLoader(
        raw_data_dir,
        epoch_type=epoch_type,
        verbose=verbose,
        random_state=seed,
        dtcwt_cfg=dtcwt_cfg,
        channel_mode=channel_mode,
        apply_ica=apply_ica,
        ica_method=ica_method,
        ica_n_components=ica_n_components,
        ica_random_state=ica_random_state,
        ica_max_iter=ica_max_iter,
    )
    records = loader.load_data()
    n = len(records)

    if channel_mode == "paper10" and loader.channel_indices is not None:
        if len(loader.channel_indices) != 10:
            raise RuntimeError(
                f"[Guard] channel_mode=paper10 expects 10 channels, "
                f"but loader selected {len(loader.channel_indices)}. Check PAPER_CHANNELS mapping."
            )

    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr = int(round(n * train_ratio))
    n_va = int(round(n * val_ratio))
    tr_idx = idx[:n_tr].tolist()
    va_idx = idx[n_tr:n_tr + n_va].tolist()
    te_idx = idx[n_tr + n_va:].tolist()

    cfg, tfm, ch = loader.dtcwt_cfg, loader._dtcwt_transform, loader.channel_indices
    return (
        KaraOnePhoneticDataset(records, ch, tfm, cfg, tr_idx, augment=augment_train,
                               feature_mode=feature_mode, ccv_repr=ccv_repr,
                               ccv_reject_alpha=ccv_reject_alpha, cov_bounded_scale=cov_bounded_scale),
        KaraOnePhoneticDataset(records, ch, tfm, cfg, va_idx, augment=False,
                               feature_mode=feature_mode, ccv_repr=ccv_repr,
                               ccv_reject_alpha=ccv_reject_alpha, cov_bounded_scale=cov_bounded_scale),
        KaraOnePhoneticDataset(records, ch, tfm, cfg, te_idx, augment=False,
                               feature_mode=feature_mode, ccv_repr=ccv_repr,
                               ccv_reject_alpha=ccv_reject_alpha, cov_bounded_scale=cov_bounded_scale),
        loader,
    )


def loso_splits(
    raw_data_dir: str,
    epoch_type: str = "thinking",
    seed: int = 42,
    verbose: bool = False,
    augment_train: bool = False,
    dtcwt_cfg: Optional[DTCWTConfig] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    channel_mode: str = "all",
    feature_mode: str = "raw",
    ccv_repr: str = "cov_bounded",
    ccv_reject_alpha: float = 0.10,
    cov_bounded_scale: float = 0.10,
    apply_ica: bool = True,
    ica_method: str = "fastica",
    ica_n_components: Optional[int] = None,
    ica_random_state: int = 97,
    ica_max_iter: int = 512,
) -> Tuple[KaraOnePhoneticDataset, KaraOnePhoneticDataset, KaraOnePhoneticDataset, KaraOneDataLoader, int]:
    """
    LOSO CV generator.

    Important fix:
        - test set: held-out subject (all trials)
        - train/val: same subjects (all except test), but split trials per subject.
          This ensures validation subjects are seen during training (unlike previous version),
          making adversarial subject validation meaningful and avoiding val loss explosion.
    """
    if channel_mode not in ("paper10", "all"):
        raise ValueError(f"Invalid channel_mode={channel_mode!r}. Must be 'paper10' or 'all'.")
    base_loader = KaraOneDataLoader(
        raw_data_dir,
        epoch_type=epoch_type,
        verbose=verbose,
        random_state=seed,
        dtcwt_cfg=dtcwt_cfg,
        channel_mode=channel_mode,
        apply_ica=apply_ica,
        ica_method=ica_method,
        ica_n_components=ica_n_components,
        ica_random_state=ica_random_state,
        ica_max_iter=ica_max_iter,
    )
    records = base_loader.load_data()
    all_sids = sorted(set(int(r["subject_id"]) for r in records))

    if channel_mode == "paper10" and base_loader.channel_indices is not None:
        if len(base_loader.channel_indices) != 10:
            raise RuntimeError(
                f"[Guard] channel_mode=paper10 expects 10 channels, "
                f"but loader selected {len(base_loader.channel_indices)}. Check PAPER_CHANNELS mapping."
            )

    cfg, tfm, ch = base_loader.dtcwt_cfg, base_loader._dtcwt_transform, base_loader.channel_indices

    for test_sid in all_sids:
        # =========================================================
        # LOSO (subject-level test) + within-train trial validation
        #
        # - test: all trials from held-out subject (test_sid)
        # - train/val: SAME subjects (all except test_sid),
        #              but split trials WITHIN each subject
        #
        # This avoids the pathological case where "val subjects are unseen",
        # which breaks adversarial subject-head validation and inflates val loss.
        # =========================================================
        rng = np.random.RandomState(seed + int(test_sid) * 1009)

        # collect indices by subject for all non-test subjects
        by_sid = {}
        for i, r in enumerate(records):
            sid = int(r["subject_id"])
            if sid == test_sid:
                continue
            by_sid.setdefault(sid, []).append(i)

        tr_idx: List[int] = []
        va_idx: List[int] = []

        # per-subject trial split (keeps subject distribution consistent)
        # val fraction is computed relative to (train+val) pool
        denom = max(1e-8, float(train_ratio + val_ratio))
        val_frac = float(val_ratio) / denom

        for sid, idxs in by_sid.items():
            idxs = list(idxs)
            rng.shuffle(idxs)
            n = len(idxs)
            if n <= 1:
                # cannot split; keep in train
                tr_idx.extend(idxs)
                continue
            n_va = int(round(n * val_frac))
            # ensure both splits non-empty when possible
            n_va = max(1, min(n - 1, n_va))
            va_idx.extend(idxs[:n_va])
            tr_idx.extend(idxs[n_va:])

        # test indices: all trials from held-out subject
        te_idx = [i for i, r in enumerate(records) if int(r["subject_id"]) == test_sid]

        yield (
            KaraOnePhoneticDataset(records, ch, tfm, cfg, tr_idx, augment=augment_train,
                                   feature_mode=feature_mode, ccv_repr=ccv_repr,
                                   ccv_reject_alpha=ccv_reject_alpha, cov_bounded_scale=cov_bounded_scale),
            KaraOnePhoneticDataset(records, ch, tfm, cfg, va_idx, augment=False,
                                   feature_mode=feature_mode, ccv_repr=ccv_repr,
                                   ccv_reject_alpha=ccv_reject_alpha, cov_bounded_scale=cov_bounded_scale),
            KaraOnePhoneticDataset(records, ch, tfm, cfg, te_idx, augment=False,
                                   feature_mode=feature_mode, ccv_repr=ccv_repr,
                                   ccv_reject_alpha=ccv_reject_alpha, cov_bounded_scale=cov_bounded_scale),
            base_loader,
            test_sid,
        )