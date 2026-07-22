#!/usr/bin/env python3
"""Multi-GPU controlled reproduction of Patch Policy's dense-token claim."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


IMAGE_SIZE = 224
PATCH_SIZE = 14
PATCH_GRID = IMAGE_SIZE // PATCH_SIZE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def render_sequences(n: int, context: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a deterministic goal-directed control dataset on CPU.

    Red is the controlled effector, green is the goal, blue/yellow are
    distractors. The previous frame exposes velocity while the label is the
    clipped displacement required to move the current effector to the goal.
    """
    gen = torch.Generator().manual_seed(seed)
    images = torch.zeros(n, context, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
    actions = torch.empty(n, 2)
    for i in range(n):
        goal = torch.rand(2, generator=gen) * 0.76 + 0.12
        current = torch.rand(2, generator=gen) * 0.76 + 0.12
        velocity = (torch.rand(2, generator=gen) - 0.5) * 0.10
        actions[i] = torch.clamp(goal - current, -0.5, 0.5) * 2.0
        for t in range(context):
            pos = current - velocity * float(context - 1 - t)
            canvas = images[i, t]
            _draw_square(canvas, pos, channel=0, radius=7, value=255)
            _draw_square(canvas, goal, channel=1, radius=8, value=255)
            # Independently positioned distractors prevent shortcut learning.
            for d in range(3):
                xy = torch.rand(2, generator=gen) * 0.88 + 0.06
                _draw_square(canvas, xy, channel=2 if d < 2 else 0, radius=4, value=110)
    return images, actions


def _draw_square(image: torch.Tensor, xy: torch.Tensor, channel: int, radius: int, value: int) -> None:
    x = int(float(xy[0]) * (IMAGE_SIZE - 1))
    y = int(float(xy[1]) * (IMAGE_SIZE - 1))
    x0, x1 = max(0, x - radius), min(IMAGE_SIZE, x + radius + 1)
    y0, y1 = max(0, y - radius), min(IMAGE_SIZE, y + radius + 1)
    image[channel, y0:y1, x0:x1] = value


class FrozenPatchProjection(nn.Module):
    """Deterministic fallback preserving local color and geometry."""

    def __init__(self, dim: int = 384):
        super().__init__()
        proj_gen = torch.Generator().manual_seed(260718236)
        projection = torch.randn(3, dim, generator=proj_gen) / math.sqrt(3)
        self.register_buffer("projection", projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(x, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        tokens = pooled.flatten(2).transpose(1, 2)
        return tokens @ self.projection


class DinoPatchEncoder(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        import timm

        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
            img_size=IMAGE_SIZE,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(x)
        # timm returns CLS + register/prefix tokens followed by the 16x16 grid.
        return features[:, -PATCH_GRID * PATCH_GRID :, :]


def prepare_encoder(requested: str) -> tuple[str, str | None]:
    if requested != "dinov2_vits14":
        return "frozen_patch_projection", None
    try:
        model = DinoPatchEncoder(pretrained=True)
        del model
        return "dinov2_vits14_lvd142m", None
    except Exception as exc:  # fallback is explicit in the evidence record
        return "frozen_patch_projection", f"{type(exc).__name__}: {exc}"


def make_encoder(resolved: str, device: torch.device) -> nn.Module:
    if resolved == "dinov2_vits14_lvd142m":
        return DinoPatchEncoder(pretrained=True).to(device)
    return FrozenPatchProjection().to(device)


@torch.inference_mode()
def encode_images(
    encoder: nn.Module,
    images: torch.Tensor,
    context: int,
    device: torch.device,
    batch_size: int = 96,
) -> torch.Tensor:
    flat = images.flatten(0, 1)
    outputs: list[torch.Tensor] = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for start in range(0, len(flat), batch_size):
        x = flat[start : start + batch_size].to(device=device, dtype=torch.float32) / 255.0
        x = (x - mean) / std
        outputs.append(encoder(x).to(dtype=torch.float16).cpu())
    encoded = torch.cat(outputs, dim=0)
    return encoded.reshape(len(images), context, encoded.shape[1], encoded.shape[2])


def attention_mask(kind: str, context: int, patches: int, device: torch.device) -> torch.Tensor | None:
    length = context * patches
    if kind == "full":
        return None
    if kind == "token_causal":
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
    if kind != "block_causal":
        raise ValueError(f"unknown attention mask: {kind}")
    frame = torch.arange(length, device=device) // patches
    return frame.unsqueeze(1) < frame.unsqueeze(0)


class PatchPolicy(nn.Module):
    def __init__(self, feature_dim: int, patches: int, cfg: dict[str, Any]):
        super().__init__()
        dim = int(cfg["embedding_dim"])
        context = int(cfg["context_frames"])
        self.patches = patches
        self.context = context
        self.attention = str(cfg["attention"])
        self.input_projection = nn.Linear(feature_dim, dim)
        self.spatial_embedding = nn.Parameter(torch.randn(1, 1, patches, dim) * 0.02)
        self.temporal_embedding = nn.Parameter(torch.randn(1, context, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(cfg["heads"]),
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(cfg["layers"]), norm=nn.LayerNorm(dim))
        self.action_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2), nn.Tanh())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(features.float())
        x = x + self.spatial_embedding + self.temporal_embedding
        x = x.flatten(1, 2)
        mask = attention_mask(self.attention, self.context, self.patches, x.device)
        x = self.transformer(x, mask=mask)
        # The last patch of the current frame is the paper's VQ-BeT readout.
        return self.action_head(x[:, -1])


@dataclass
class SeedResult:
    seed: int
    val_mse: float
    success_rate: float
    cosine: float
    best_epoch: int
    train_seconds: float


def run_seed(rank: int, seed: int, cfg: dict[str, Any], resolved_encoder: str, queue: mp.Queue) -> None:
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        seed_everything(seed)
        print(f"SEED_START seed={seed} gpu={rank} encoder={resolved_encoder}", flush=True)
        train_images, train_actions = render_sequences(int(cfg["train_samples"]), int(cfg["context_frames"]), seed)
        val_images, val_actions = render_sequences(int(cfg["val_samples"]), int(cfg["context_frames"]), seed + 100_000)
        encoder = make_encoder(resolved_encoder, device)
        train_features = encode_images(encoder, train_images, int(cfg["context_frames"]), device)
        val_features = encode_images(encoder, val_images, int(cfg["context_frames"]), device)
        del encoder, train_images, val_images
        torch.cuda.empty_cache()

        if cfg["representation"] == "global_avg":
            train_features = train_features.mean(dim=2, keepdim=True)
            val_features = val_features.mean(dim=2, keepdim=True)
        elif cfg["representation"] == "spatial_4x4":
            train_features = spatial_pool(train_features, output_grid=4)
            val_features = spatial_pool(val_features, output_grid=4)
        elif cfg["representation"] != "dense":
            raise ValueError(f"unknown representation: {cfg['representation']}")

        model = PatchPolicy(train_features.shape[-1], train_features.shape[2], cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        train_loader = DataLoader(
            TensorDataset(train_features, train_actions),
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            pin_memory=True,
        )
        val_loader = DataLoader(TensorDataset(val_features, val_actions), batch_size=128, pin_memory=True)
        best: SeedResult | None = None
        started = time.time()
        for epoch in range(1, int(cfg["epochs"]) + 1):
            model.train()
            loss_sum = 0.0
            for features, target in train_loader:
                features = features.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                prediction = model(features)
                loss = F.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss) * len(features)
            metrics = evaluate(model, val_loader, device, float(cfg["success_threshold"]))
            result = SeedResult(seed, metrics["mse"], metrics["success"], metrics["cosine"], epoch, time.time() - started)
            if best is None or result.val_mse < best.val_mse:
                best = result
            print(
                f"EPOCH seed={seed} epoch={epoch} train_mse={loss_sum / len(train_loader.dataset):.6f} "
                f"val_mse={metrics['mse']:.6f} success={metrics['success']:.4f} cosine={metrics['cosine']:.4f}",
                flush=True,
            )
        assert best is not None
        print(f"SEED_RESULT {json.dumps(best.__dict__, sort_keys=True)}", flush=True)
        queue.put({"ok": True, "result": best.__dict__})
    except Exception as exc:
        import traceback

        traceback.print_exc()
        queue.put({"ok": False, "seed": seed, "error": f"{type(exc).__name__}: {exc}"})


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, float]:
    model.eval()
    squared = 0.0
    success = 0.0
    cosine = 0.0
    count = 0
    for features, target in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction = model(features)
        error = prediction - target
        squared += float(error.square().sum())
        success += float((error.norm(dim=-1) < threshold).sum())
        cosine += float(F.cosine_similarity(prediction, target, dim=-1).sum())
        count += len(features)
    return {"mse": squared / (count * 2), "success": success / count, "cosine": cosine / count}


def spatial_pool(features: torch.Tensor, output_grid: int) -> torch.Tensor:
    """Average-pool a square patch grid while preserving coarse layout."""
    n, context, patches, dim = features.shape
    input_grid = math.isqrt(patches)
    if input_grid * input_grid != patches or input_grid % output_grid:
        raise ValueError(f"cannot pool {patches} patches to {output_grid}x{output_grid}")
    grid = features.reshape(n * context, input_grid, input_grid, dim).permute(0, 3, 1, 2).float()
    grid = F.avg_pool2d(grid, kernel_size=input_grid // output_grid, stride=input_grid // output_grid)
    return grid.permute(0, 2, 3, 1).reshape(n, context, output_grid * output_grid, dim).half()


def summarize(cfg: dict[str, Any], resolved: str, fallback_error: str | None, results: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate(key: str) -> dict[str, float]:
        values = [float(result[key]) for result in results]
        return {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "experiment": cfg["experiment"],
        "representation": cfg["representation"],
        "attention": cfg["attention"],
        "encoder_requested": cfg["encoder"],
        "encoder_resolved": resolved,
        "encoder_fallback_error": fallback_error,
        "num_seeds": len(results),
        "val_mse": aggregate("val_mse"),
        "success_rate": aggregate("success_rate"),
        "cosine": aggregate("cosine"),
        "seeds": results,
        "paper_reference": {"dense_push_t_vqbet": 0.69, "webssl_avg_pool_push_t_vqbet": 0.54},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    visible = torch.cuda.device_count()
    requested_seeds = int(cfg["seeds"])
    print(f"CONFIG {json.dumps(cfg, sort_keys=True)}", flush=True)
    print(f"SYSTEM torch={torch.__version__} cuda={torch.version.cuda} visible_gpus={visible}", flush=True)
    if visible < requested_seeds:
        raise RuntimeError(f"requires {requested_seeds} GPUs but only {visible} are visible")
    resolved, fallback_error = prepare_encoder(str(cfg["encoder"]))
    print(f"ENCODER requested={cfg['encoder']} resolved={resolved} fallback_error={fallback_error!r}", flush=True)

    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    processes = [
        context.Process(target=run_seed, args=(rank, 10_000 + rank, cfg, resolved, queue))
        for rank in range(requested_seeds)
    ]
    for process in processes:
        process.start()
    messages = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    failures = [message for message in messages if not message["ok"]]
    if failures:
        print(f"ORX_FAILURES {json.dumps(failures, sort_keys=True)}", flush=True)
        raise RuntimeError(f"{len(failures)} seed workers failed")
    results = sorted((message["result"] for message in messages), key=lambda x: x["seed"])
    summary = summarize(cfg, resolved, fallback_error, results)
    print(f"ORX_RESULT {json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
