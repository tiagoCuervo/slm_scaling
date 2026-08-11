from __future__ import annotations

import numpy as np
import torch


def load_codehifigan(
    device: str | torch.device = "cpu",
    vocab_size: int = 500,
    *,
    cache_dir: str | None = None,
):
    """Download verified official assets and load the native unit vocoder."""
    if vocab_size != 500:
        raise ValueError("the released mHuBERT CodeHiFiGAN vocoder requires 500 units")
    from .vocoder import load_vocoder

    return load_vocoder(device, cache_dir=cache_dir)


@torch.inference_mode()
def decode_units(codes, vocoder, *, vocab_size: int = 500) -> np.ndarray:
    tokens = torch.as_tensor(codes, dtype=torch.long, device=next(vocoder.parameters()).device).reshape(1, -1)
    tokens = tokens[(tokens >= 0) & (tokens < vocab_size)].reshape(1, -1)
    if not tokens.numel():
        raise ValueError("no decodable HuBERT units")
    waveform = vocoder(tokens, dur_prediction=True)
    return waveform.squeeze().float().cpu().numpy()
