"""Generic trainer used by forward and inverse pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from tqdm import trange


@dataclass(frozen=True)
class TrainerConfig:
    """Runtime settings for optimization loops."""

    epochs: int = 1000
    mixed_precision: bool = False
    grad_clip_norm: float | None = None
    log_every: int = 20

    @classmethod
    def from_config(cls, config: dict) -> "TrainerConfig":
        source = config.get("training", config)
        clip = source.get("grad_clip_norm", None)
        return cls(
            epochs=int(source.get("epochs", 1000)),
            mixed_precision=bool(source.get("mixed_precision", False)),
            grad_clip_norm=None if clip is None else float(clip),
            log_every=int(source.get("log_every", 20)),
        )


class Trainer:
    """Small training engine around an optimization step callback."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        logger=None,
        config: TrainerConfig | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.config = config or TrainerConfig()

    def train(
        self,
        step_fn: Callable[[int], tuple[Tensor, dict[str, float]]],
        on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        """Run optimization loop and return per-epoch metrics."""
        history: list[dict[str, float]] = []

        scaler = torch.cuda.amp.GradScaler(
            enabled=self.config.mixed_precision and torch.cuda.is_available()
        )
        is_lbfgs = isinstance(self.optimizer, torch.optim.LBFGS)

        if is_lbfgs and scaler.is_enabled():
            raise ValueError("LBFGS is not supported with mixed_precision=True")

        for epoch in trange(1, self.config.epochs + 1, desc="training", leave=False):
            if is_lbfgs:
                latest_metrics: dict[str, float] = {}
                latest_loss: Tensor | None = None

                def closure() -> Tensor:
                    nonlocal latest_loss, latest_metrics
                    self.optimizer.zero_grad(set_to_none=True)
                    loss_local, metrics_local = step_fn(epoch)

                    if not torch.isfinite(loss_local):
                        raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

                    loss_local.backward()
                    if self.config.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config.grad_clip_norm,
                        )

                    latest_loss = loss_local
                    latest_metrics = dict(metrics_local)
                    return loss_local

                self.optimizer.step(closure)
                if latest_loss is None:
                    raise RuntimeError("LBFGS closure did not produce a valid loss")

                loss = latest_loss
                metrics = latest_metrics
            else:
                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(
                    enabled=self.config.mixed_precision and torch.cuda.is_available()
                ):
                    loss, metrics = step_fn(epoch)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if self.config.grad_clip_norm is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config.grad_clip_norm,
                        )
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.config.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config.grad_clip_norm,
                        )
                    self.optimizer.step()

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss.detach().item())
                else:
                    self.scheduler.step()

            metrics_out = dict(metrics)
            metrics_out.setdefault("total", float(loss.detach().cpu().item()))
            metrics_out["epoch"] = float(epoch)
            history.append(metrics_out)

            if self.logger and (epoch == 1 or epoch % self.config.log_every == 0):
                self.logger.info(
                    "Epoch %d | total=%.6f",
                    epoch,
                    metrics_out["total"],
                )

            if on_epoch_end is not None:
                on_epoch_end(epoch, metrics_out)

        return history
