"""Minimal mHuBERT CodeHiFiGAN inference model.

The architecture is adapted from Meta's MIT-licensed textlesslib CodeHiFiGAN
implementation and the original MIT-licensed HiFi-GAN generator. Training-only
components are intentionally omitted.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm


ASSETS = {
    "checkpoint": {
        "name": "hifigan_lj_mhubert_base_25hz.pt",
        "url": (
            "https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/"
            "hifigan_lj_mhubert_base_25hz.pt"
        ),
        "sha256": "d88224e95c501e2cd59a6e4014753169cfe060fb7ade3cc0da03c809fef73b79",
    },
    "config": {
        "name": "hifigan_lj_mhubert_base_25hz_config.json",
        "url": (
            "https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/"
            "hifigan_lj_mhubert_base_25hz_config.json"
        ),
        "sha256": "116dc39be4970cd393e562acd0a0ec70a86aaba7e9ca469b8b836cb3fe46afc7",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download_asset(kind: str, cache_dir: str | Path | None) -> Path:
    spec = ASSETS[kind]
    root = Path(cache_dir) if cache_dir is not None else Path(torch.hub.get_dir()) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / spec["name"]
    if destination.exists() and _sha256(destination) == spec["sha256"]:
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urlopen(spec["url"], timeout=60) as source, partial.open("wb") as output:
            while chunk := source.read(8 << 20):
                output.write(chunk)
        if _sha256(partial) != spec["sha256"]:
            raise RuntimeError(f"checksum mismatch for {spec['name']}")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: list[int]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=value,
                    padding=_padding(kernel_size, value),
                )
            )
            for value in dilation
        )
        self.convs2 = nn.ModuleList(
            weight_norm(
                nn.Conv1d(channels, channels, kernel_size, padding=_padding(kernel_size))
            )
            for _ in dilation
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = conv1(F.leaky_relu(inputs, 0.1))
            residual = conv2(F.leaky_relu(residual, 0.1))
            inputs = inputs + residual
        return inputs

    def remove_weight_norm(self) -> None:
        for layer in (*self.convs1, *self.convs2):
            remove_weight_norm(layer)


class _DurationPredictor(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dim = config["encoder_embed_dim"]
        hidden = config["var_pred_hidden_dim"]
        kernel = config["var_pred_kernel_size"]
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel, padding=(kernel - 1) // 2), nn.ReLU()
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel, padding=(kernel - 1) // 2), nn.ReLU()
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = config["var_pred_dropout"]
        self.projection = nn.Linear(hidden, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv1(inputs.transpose(1, 2)).transpose(1, 2)
        inputs = F.dropout(self.norm1(inputs), self.dropout, self.training)
        inputs = self.conv2(inputs.transpose(1, 2)).transpose(1, 2)
        inputs = F.dropout(self.norm2(inputs), self.dropout, self.training)
        return self.projection(inputs).squeeze(2)


class _CodeGenerator(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embedding = nn.Embedding(config["num_embeddings"], config["embedding_dim"])
        channels = config["upsample_initial_channel"]
        self.pre = weight_norm(nn.Conv1d(config["model_in_dim"], channels, 7, padding=3))
        self.upsamples = nn.ModuleList()
        self.residuals = nn.ModuleList()
        for index, (rate, kernel) in enumerate(
            zip(config["upsample_rates"], config["upsample_kernel_sizes"])
        ):
            input_channels = channels // (2**index)
            output_channels = channels // (2 ** (index + 1))
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        input_channels,
                        output_channels,
                        kernel,
                        rate,
                        padding=(kernel - rate) // 2,
                    )
                )
            )
            self.residuals.extend(
                _ResidualBlock(output_channels, size, dilation)
                for size, dilation in zip(
                    config["resblock_kernel_sizes"], config["resblock_dilation_sizes"]
                )
            )
        self.num_kernels = len(config["resblock_kernel_sizes"])
        self.post = weight_norm(nn.Conv1d(output_channels, 1, 7, padding=3))
        self.duration = _DurationPredictor(config["dur_predictor_params"])

    def forward(self, code: torch.Tensor, *, dur_prediction: bool) -> torch.Tensor:
        inputs = self.embedding(code).transpose(1, 2)
        if dur_prediction:
            if inputs.size(0) != 1:
                raise ValueError("duration prediction supports one sequence at a time")
            log_duration = self.duration(inputs.transpose(1, 2))
            duration = torch.clamp(torch.round(torch.exp(log_duration) - 1).long(), min=1)
            inputs = torch.repeat_interleave(inputs, duration.reshape(-1), dim=2)
        inputs = self.pre(inputs)
        for index, upsample in enumerate(self.upsamples):
            inputs = upsample(F.leaky_relu(inputs, 0.1))
            start = index * self.num_kernels
            combined = self.residuals[start](inputs)
            for branch in range(1, self.num_kernels):
                combined = combined + self.residuals[start + branch](inputs)
            inputs = combined / self.num_kernels
        # The released generator used PyTorch's default 0.01 slope at this
        # final activation (the upsampling and residual stages use 0.1).
        return torch.tanh(self.post(F.leaky_relu(inputs)))

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self.pre)
        remove_weight_norm(self.post)
        for layer in self.upsamples:
            remove_weight_norm(layer)
        for layer in self.residuals:
            layer.remove_weight_norm()


class CodeHiFiGANVocoder(nn.Module):
    def __init__(self, checkpoint: Path, config: Path):
        super().__init__()
        self.config = json.loads(config.read_text())
        self.generator = _CodeGenerator(self.config)
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        replacements = {
            "dict.": "embedding.",
            "conv_pre.": "pre.",
            "ups.": "upsamples.",
            "resblocks.": "residuals.",
            "conv_post.": "post.",
            "dur_predictor.conv1.": "duration.conv1.",
            "dur_predictor.ln1.": "duration.norm1.",
            "dur_predictor.conv2.": "duration.conv2.",
            "dur_predictor.ln2.": "duration.norm2.",
            "dur_predictor.proj.": "duration.projection.",
        }
        remapped = {}
        for name, tensor in state["generator"].items():
            for old, new in replacements.items():
                if name.startswith(old):
                    name = new + name[len(old):]
                    break
            remapped[name] = tensor
        self.generator.load_state_dict(remapped)
        self.generator.remove_weight_norm()
        self.eval()

    @property
    def output_sample_rate(self) -> int:
        return int(self.config["sampling_rate"])

    def forward(self, code: torch.Tensor, *, dur_prediction: bool = True) -> torch.Tensor:
        code = code[(code >= 0) & (code < self.config["num_embeddings"])].reshape(1, -1)
        if code.numel() == 0:
            raise ValueError("no valid units to decode")
        return self.generator(code, dur_prediction=dur_prediction).squeeze()


def load_vocoder(
    device: str | torch.device = "cpu", *, cache_dir: str | Path | None = None
) -> CodeHiFiGANVocoder:
    checkpoint = _download_asset("checkpoint", cache_dir)
    config = _download_asset("config", cache_dir)
    return CodeHiFiGANVocoder(checkpoint, config).to(device).eval()
