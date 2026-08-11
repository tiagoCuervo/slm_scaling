from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def load_audio(path: str | Path, *, offset: float = 0.0, duration: float | None = None, target_rate: int = 16000):
    """Load a mono segment with SoundFile and resample it once when needed."""
    import soundfile as sf

    info = sf.info(path)
    start = round(offset * info.samplerate)
    frames = -1 if duration is None else round(duration * info.samplerate)
    waveform, sample_rate = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != target_rate:
        try:
            from torchaudio.functional import resample
        except ImportError as exc:
            raise ImportError("resampling requires torchaudio; source audio is not 16 kHz") from exc
        waveform = resample(
            torch.from_numpy(np.asarray(waveform)).view(1, -1),
            int(sample_rate),
            target_rate,
        ).view(-1).numpy()
    return np.asarray(waveform, dtype=np.float32), target_rate


def collapse_codes(codes: np.ndarray, eos_token: int) -> np.ndarray:
    """Append EOS and collapse consecutive units."""
    codes = np.asarray(codes)
    if codes.ndim != 1:
        raise ValueError(f"expected one unit stream, got shape {codes.shape}")
    codes = np.concatenate((codes, np.asarray([eos_token], dtype=codes.dtype)))
    keep = np.ones(len(codes), dtype=bool)
    keep[1:] = codes[1:] != codes[:-1]
    return codes[keep].astype(np.int16)


class HubertKMeansTokenizer:
    """Hugging Face mHuBERT features followed by K-means quantization."""

    def __init__(
        self,
        model: str,
        kmeans: str,
        revision: str,
        layer: int = 11,
        device: str = "cuda",
        max_seconds: int = 30,
        normalize: bool | None = None,
    ):
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("install the slm-data hubert dependencies") from exc
        try:
            from transformers import AutoFeatureExtractor, HubertModel
        except ImportError as exc:
            raise ImportError("install the slm-data hubert dependencies") from exc
        extractor = AutoFeatureExtractor.from_pretrained(model, revision=revision)
        backend_normalize = bool(getattr(extractor, "do_normalize", False))
        self.model = HubertModel.from_pretrained(model, revision=revision).to(device).eval()
        self.normalize = backend_normalize if normalize is None else bool(normalize)
        self.centers = torch.from_numpy(joblib.load(kmeans).cluster_centers_).float().to(device)
        if len(self.centers) > np.iinfo(np.int16).max - 1:
            raise ValueError("K-means vocabulary plus EOS/PAD must fit in int16")
        self.center_norms = self.centers.square().sum(1)
        self.layer = layer
        self.device = device
        self.max_samples = 16000 * max_seconds
        self.eos_token = len(self.centers)
        self.pad_token = self.eos_token + 1

    def _features(self, chunk: torch.Tensor) -> torch.Tensor:
        output = self.model(
            input_values=chunk.unsqueeze(0),
            output_hidden_states=True,
            return_dict=True,
        )
        if output.hidden_states is None or self.layer >= len(output.hidden_states):
            raise ValueError(
                f"mHuBERT layer {self.layer} is unavailable; "
                f"model returned {len(output.hidden_states or ())} hidden states"
            )
        return output.hidden_states[self.layer][0]

    @torch.inference_mode()
    def encode_uncollapsed(self, waveform: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(waveform, dtype=torch.float32, device=self.device)
        if self.normalize:
            x = F.layer_norm(x, x.shape)
        features = []
        for start in range(0, x.numel(), self.max_samples):
            chunk = x[start:start + self.max_samples]
            if chunk.numel() < 720:
                continue
            features.append(self._features(chunk))
        if not features:
            raise ValueError("audio is too short for HuBERT")
        feature = torch.cat(features)
        feature_norm = feature.square().sum(1, keepdim=True)
        distance = feature_norm - 2 * feature @ self.centers.T + self.center_norms
        return distance.argmin(1).cpu().numpy().astype(np.int16)

    @torch.inference_mode()
    def encode(self, waveform: np.ndarray) -> np.ndarray:
        return collapse_codes(self.encode_uncollapsed(waveform), self.eos_token)
