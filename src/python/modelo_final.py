import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import soundfile as sf

# CONFIGURACIÓN GLOBAL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MANIFEST_TRAIN = r"C:\Users\KEVIN SORIANO\Downloads\espectrogramas_for_finetune\manifest_train.csv"
MANIFEST_VAL   = r"C:\Users\KEVIN SORIANO\Downloads\espectrogramas_for_finetune\manifest_val.csv"
MANIFEST_TEST  = r"C:\Users\KEVIN SORIANO\Downloads\espectrogramas_for_finetune\manifest_test.csv"

# Espectrograma
N_MELS = 192
TIME_STEPS = 256
CHANNELS = 3 

WAV_SAMPLES = 48000  

# Entrenamiento
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-4
USE_AUG = True

# AUGMENTATIONS

def augment_wave(x):

    if random.random() < 0.5:
        noise = np.random.randn(len(x)) * 0.01
        x = x + noise

    if random.random() < 0.5:
        steps = random.uniform(-2, 2)
        x = librosa.effects.pitch_shift(x, sr=16000, n_steps=steps)

    if random.random() < 0.5:
        rate = random.uniform(0.8, 1.2)
        x = librosa.resample(x, orig_sr=16000, target_sr=int(16000 * rate))

    return x

# DATASET HÍBRIDO 
class HybridDataset(Dataset):
    def __init__(self, manifest_csv, augment=False):
        self.df = pd.read_csv(manifest_csv)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def extract_features(self, audio):
        """Jitter, shimmer, energy, spectral flux, etc."""
        feats = []

        feats.append(np.mean(audio ** 2))


        S = np.abs(librosa.stft(audio))
        flux = np.sum(np.diff(S, axis=1) ** 2)
        feats.append(flux)

        f0, _, _ = librosa.pyin(audio, fmin=80, fmax=350)
        feats.append(np.nanmean(f0))

        feats = [0 if np.isnan(v) else v for v in feats]
        return np.array(feats, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # cargar espectrograma 
        spec = np.load(row["npy_path"]).astype(np.float32)
        spec = np.transpose(spec, (2, 0, 1))

        # cargar waveform
        wav, _ = librosa.load(row["orig_audio"], sr=22050)

        if self.augment:
            wav = augment_wave(wav)

        if len(wav) < WAV_SAMPLES:
            wav = np.pad(wav, (0, WAV_SAMPLES - len(wav)))
        else:
            wav = wav[:WAV_SAMPLES]

        feats = self.extract_features(wav)

        return (
            torch.tensor(spec, dtype=torch.float32),
            torch.tensor(wav, dtype=torch.float32),
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(row["label"], dtype=torch.long)
        )


# MODELO PRINCIPAL
class CNN2D(nn.Module):
    """Procesa espectrogramas."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )

        self.flat_dim = 128 * (N_MELS // 4) * (TIME_STEPS // 4)
        self.fc = nn.Linear(self.flat_dim, 256)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class CNN1D(nn.Module):
    """Procesa waveform bruto."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 64, 9, stride=2, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        self.fc = nn.Linear(64*64, 128)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.att(x), dim=1)
        return torch.sum(w * x, dim=1)


class HybridAntiDeepfake(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch_spec = CNN2D()
        self.branch_wave = CNN1D()
        self.branch_feat = nn.Linear(3, 64)

        self.lstm = nn.LSTM(
            input_size=256 + 128 + 64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.att = Attention(256)
        self.fc = nn.Linear(256, 2)

    def forward(self, spec, wav, feat):
        b = spec.size(0)

        x1 = self.branch_spec(spec)
        x2 = self.branch_wave(wav)
        x3 = torch.relu(self.branch_feat(feat))

        x = torch.cat([x1, x2, x3], dim=1)
        x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)
        att = self.att(lstm_out)
        return self.fc(att)


# ENTRENAMIENTO 

def train_loop():

    train_set = HybridDataset(MANIFEST_TRAIN, augment=True)
    val_set   = HybridDataset(MANIFEST_VAL, augment=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = HybridAntiDeepfake().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total = 0
        correct = 0

        for spec, wav, feat, label in train_loader:
            spec, wav, feat, label = spec.to(DEVICE), wav.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)

            opt.zero_grad()
            out = model(spec, wav, feat)
            loss = crit(out, label)
            loss.backward()
            opt.step()

            pred = out.argmax(1)
            total += label.size(0)
            correct += (pred == label).sum().item()

        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Acc: {acc:.4f}")

    torch.save(model.state_dict(), "anti_deepfake_cnn_lstm.pt")
    print("Modelo guardado → anti_deepfake_cnn_lstm.pt")


if __name__ == "__main__":
    train_loop()
