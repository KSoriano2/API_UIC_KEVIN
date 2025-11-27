import sys
import json
import os
import torch
import librosa
import numpy as np
import soundfile as sf
from modelo_final import HybridAntiDeepfake  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


N_MELS = 192
TIME_STEPS = 256
CHANNELS = 3
WAV_SAMPLES = 48000  

CLASSES = ["REAL", "DEEPFAKE"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "anti_deepfake_cnn_lstm.pt")

def load_audio(path):
    wav, sr = sf.read(path)

    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    if sr != 22050:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=22050)

    if len(wav) < WAV_SAMPLES:
        wav = np.pad(wav, (0, WAV_SAMPLES - len(wav)))
    else:
        wav = wav[:WAV_SAMPLES]

    return wav.astype(np.float32)


def compute_melspec(wav):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=22050, n_mels=N_MELS, hop_length=256
    )

    mel = librosa.power_to_db(mel, ref=np.max)
    delta = librosa.feature.delta(mel)
    delta2 = librosa.feature.delta(mel, order=2)

    melspec = np.stack([mel, delta, delta2], axis=0)

    if melspec.shape[2] < TIME_STEPS:
        pad = TIME_STEPS - melspec.shape[2]
        melspec = np.pad(melspec, ((0, 0), (0, 0), (0, pad)))
    else:
        melspec = melspec[:, :, :TIME_STEPS]

    return melspec.astype(np.float32)


def compute_features(wav):
    feats = []

    feats.append(np.mean(wav ** 2)) 

    S = np.abs(librosa.stft(wav))
    flux = np.sum(np.diff(S, axis=1) ** 2)
    feats.append(flux)

    f0, _, _ = librosa.pyin(wav, fmin=80, fmax=350)
    feats.append(np.nanmean(f0) if not np.isnan(np.nanmean(f0)) else 0)

    feats = [0 if np.isnan(f) else f for f in feats]

    return np.array(feats, dtype=np.float32)

# CARGA DEL MODELO
model = HybridAntiDeepfake().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# PREDICCIÓN 
def predict(audio_path):
    wav = load_audio(audio_path)
    melspec = compute_melspec(wav)
    feats = compute_features(wav)

    wav_t = torch.tensor(wav).float().unsqueeze(0).to(DEVICE)
    spec_t = torch.tensor(melspec).float().unsqueeze(0).to(DEVICE)
    feats_t = torch.tensor(feats).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(spec_t, wav_t, feats_t)
        prob = torch.softmax(out, dim=1)[0]
        pred = torch.argmax(prob).item()

    return {
        "prediccion": CLASSES[pred],
        "prob_real": float(prob[0]),
        "prob_fake": float(prob[1])
    }

if __name__ == "__main__":
    audio_path = sys.argv[1]
    resultado = predict(audio_path)
    print(json.dumps(resultado))
