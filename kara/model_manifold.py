# model_manifold.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ============================================================
# Gradient Reversal Layer
# ============================================================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_, device=x.device))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_t,) = ctx.saved_tensors
        lambda_ = float(lambda_t.item())
        return -lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, float(lambda_))


# ============================================================
# Spatial CNN (CCV → spatial features)
# ============================================================
class SpatialCNN(nn.Module):
    """
    Spatial CNN over CCV/CCV-mask.

    Input : corr [B, C, C], corr_mask [B, C, C]
    Output: [B, 128]

    Notes:
      - We infer post-pooling flatten dim dynamically (channel-count agnostic).
      - Requires C >= 4. Prefer C multiple-of-4 because we apply MaxPool2d(2) twice.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        if n_channels < 4:
            raise ValueError(f"SpatialCNN requires n_channels>=4, got {n_channels}")
        self.n_channels = int(n_channels)

        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        with torch.no_grad():
            dummy = torch.zeros(1, 2, self.n_channels, self.n_channels)
            y = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            y = self.pool2(F.relu(self.bn2(self.conv2(y))))
            flat_dim = int(y.flatten(1).shape[1])

        self.fc1 = nn.Linear(flat_dim, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, corr: torch.Tensor, corr_mask: torch.Tensor) -> torch.Tensor:
        x = torch.stack([corr, corr_mask], dim=1)  # [B, 2, C, C]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.drop1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn_fc2(self.fc2(x))))
        return x  # [B, 128]


# ============================================================
# Temporal LSTM
# ============================================================
class TemporalLSTM(nn.Module):
    """
    Input : [B, T, C]
    Output: [B, 1024]
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim,
            256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.drop_lstm_out = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)

        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, seq_x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        out, _ = self.lstm(seq_x)  # [B, T, 256]

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long() - 1
            max_len = out.size(1) - 1
            lengths = lengths.clamp(min=0, max=max_len)
            H = out.size(2)
            idx = lengths.view(-1, 1, 1).expand(-1, 1, H)
            last_out = out.gather(1, idx).squeeze(1)
        else:
            last_out = out[:, -1, :]

        last_out = self.drop_lstm_out(last_out)
        x = self.drop1(F.relu(self.bn_fc1(self.fc1(last_out))))
        x = self.drop2(F.relu(self.bn_fc2(self.fc2(x))))
        return x  # [B, 1024]


# ============================================================
# EEGPhonologicalManifoldNet
# ============================================================
class EEGPhonologicalManifoldNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_subjects: int = 14,
        n_tasks: int = 5,
        latent_dim: int = 32,
        grl_lambda: float = 1.0,
        use_nonlinear_clf: bool = False,
    ):
        super().__init__()
        self.grl_lambda = float(grl_lambda)
        self.latent_dim = int(latent_dim)

        self.cnn = SpatialCNN(n_channels=n_channels)
        self.lstm = TemporalLSTM(in_dim=n_channels)

        merged_dim = 128 + 1024

        # encoder
        self.enc_fc1 = nn.Linear(merged_dim, 512)
        self.bn_enc1 = nn.BatchNorm1d(512)
        self.enc_fc2 = nn.Linear(512, 128)
        self.bn_enc2 = nn.BatchNorm1d(128)
        self.enc_fc3 = nn.Linear(128, latent_dim)

        # decoder (for DAE-style reconstruction of merged features)
        self.dec_fc1 = nn.Linear(latent_dim, 128)
        self.bn_dec1 = nn.BatchNorm1d(128)
        self.dec_fc2 = nn.Linear(128, 512)
        self.bn_dec2 = nn.BatchNorm1d(512)
        self.dec_fc3 = nn.Linear(512, merged_dim)

        self.drop = nn.Dropout(0.3)

        if use_nonlinear_clf:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_tasks),
            )
        else:
            self.classifier = nn.Linear(latent_dim, n_tasks)

        # subject adversarial head
        self.subject_clf = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim, n_subjects),
        )

    def set_grl_lambda(self, value: float):
        """
        Allows training script to schedule GRL strength across epochs.
        Example: linear warmup from 0 -> target in first N epochs.
        """
        self.grl_lambda = float(value)

    def encode(self, seq_x, corr, corr_mask, attention_mask=None):
        cnn_feat = self.cnn(corr, corr_mask)
        lstm_feat = self.lstm(seq_x, attention_mask)
        merged = torch.cat([cnn_feat, lstm_feat], dim=1)

        merged_in = merged
        if self.training:
            merged_in = F.dropout(merged_in, p=0.10, training=True)
            merged_in = merged_in + 0.01 * torch.randn_like(merged_in)
        z = self.drop(F.relu(self.bn_enc1(self.enc_fc1(merged_in))))
        z = self.drop(F.relu(self.bn_enc2(self.enc_fc2(z))))
        latent = self.enc_fc3(z)

        h = self.drop(F.relu(self.bn_dec1(self.dec_fc1(latent))))
        h = self.drop(F.relu(self.bn_dec2(self.dec_fc2(h))))
        merged_hat = self.dec_fc3(h)

        return latent, merged, merged_hat

    def forward(self, seq_x, corr, corr_mask, attention_mask=None):
        latent, merged, merged_hat = self.encode(seq_x, corr, corr_mask, attention_mask)
        logits = self.classifier(latent)
        rev = grad_reverse(latent, self.grl_lambda)
        subject_logits = self.subject_clf(rev)
        return logits, latent, subject_logits, merged, merged_hat