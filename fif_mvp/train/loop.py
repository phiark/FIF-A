"""Training loop and evaluation pipelines."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fif_mvp.config import ExperimentConfig
from fif_mvp.models import ModelOutput
from fif_mvp.train import metrics as metrics_lib
from fif_mvp.utils.logging import setup_file_logger
from fif_mvp.utils.timer import Timer


class Trainer:
    """Encapsulates optimization + evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: ExperimentConfig,
        save_dir: Path,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.criterion = nn.CrossEntropyLoss()
        self.logger = setup_file_logger(save_dir / "train_log.txt")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.step_times: List[float] = []
        # New torch.amp API (only for CUDA)
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if (self.config.use_amp and device.type == "cuda")
            else None
        )
        self.rank = getattr(config, "rank", 0)
        self.world_size = getattr(config, "world_size", 1)
        self.energy_reg_weight = self.config.energy_reg_weight
        self.energy_alerts: List[Dict[str, Any]] = []

    def _build_scheduler(
        self, total_steps: int
    ) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
        warmup = self.config.optimization.warmup_steps
        if warmup <= 0:
            return None

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / max(1, warmup)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def fit(self, loaders: Dict[str, DataLoader]) -> Dict[str, float]:
        train_loader = loaders["train"]
        total_steps = self.config.optimization.max_steps or (
            len(train_loader) * self.config.optimization.epochs
        )
        self.scheduler = self._build_scheduler(total_steps)

        metrics_rows: List[Dict] = []
        energy_rows: List[Dict] = []
        global_step = 0

        with Timer() as timer:
            for epoch in range(1, self.config.optimization.epochs + 1):
                epoch_reg_weight = self.energy_reg_weight
                train_metrics = self._run_epoch(
                    train_loader, epoch, global_step, total_steps
                )
                global_step = train_metrics["global_step"]
                train_alert = self._check_energy_alert("train", epoch, train_metrics)
                metrics_rows.append(
                    {
                        "epoch": epoch,
                        "split": "train",
                        "loss": train_metrics["loss"],
                        "acc": train_metrics["acc"],
                        "macro_f1": train_metrics["f1"],
                        "ece": train_metrics["ece"],
                        "energy_mean": train_metrics["energy"],
                        "energy_log_mean": train_metrics["energy_log"],
                        "energy_std": train_metrics["energy_std"],
                        "energy_p90": train_metrics["energy_p90"],
                        "energy_alert": int(train_alert),
                        "energy_reg_weight": epoch_reg_weight,
                    }
                )
                energy_rows.append(
                    {
                        "epoch": epoch,
                        "split": "train",
                        "energy_mean": train_metrics["energy"],
                        "energy_log_mean": train_metrics["energy_log"],
                        "energy_std": train_metrics["energy_std"],
                        "energy_p90": train_metrics["energy_p90"],
                        "energy_alert": int(train_alert),
                        "energy_reg_weight": epoch_reg_weight,
                    }
                )

                val_metrics = self.evaluate(
                    loaders["validation"], split="validation", epoch=epoch
                )
                val_alert = self._check_energy_alert("validation", epoch, val_metrics)
                metrics_rows.append(
                    {
                        "epoch": epoch,
                        "split": "validation",
                        **val_metrics,
                        "energy_alert": int(val_alert),
                        "energy_reg_weight": epoch_reg_weight,
                    }
                )
                energy_rows.append(
                    {
                        "epoch": epoch,
                        "split": "validation",
                        "energy_mean": val_metrics["energy_mean"],
                        "energy_log_mean": val_metrics["energy_log_mean"],
                        "energy_std": val_metrics["energy_std"],
                        "energy_p90": val_metrics["energy_p90"],
                        "energy_alert": int(val_alert),
                        "energy_reg_weight": epoch_reg_weight,
                    }
                )

                self._maybe_backoff_energy_reg(train_metrics["energy_std"], epoch=epoch)

                if global_step >= total_steps:
                    break

        test_metrics = self.evaluate(
            loaders["test"],
            split="test",
            record_confusion=True,
            compute_correlation=True,
        )
        test_alert = self._check_energy_alert("test", None, test_metrics)
        metrics_rows.append(
            {
                "epoch": 0,
                "split": "test",
                **test_metrics,
                "energy_alert": int(test_alert),
                "energy_reg_weight": self.energy_reg_weight,
            }
        )
        energy_rows.append(
            {
                "epoch": 0,
                "split": "test",
                "energy_mean": test_metrics["energy_mean"],
                "energy_log_mean": test_metrics["energy_log_mean"],
                "energy_std": test_metrics["energy_std"],
                "energy_p90": test_metrics["energy_p90"],
                "energy_alert": int(test_alert),
                "energy_reg_weight": self.energy_reg_weight,
            }
        )

        if self.rank == 0:
            pd.DataFrame(metrics_rows).to_csv(
                self.save_dir / "metrics_epoch.csv", index=False
            )
            pd.DataFrame(energy_rows).to_csv(
                self.save_dir / "energy_epoch.csv", index=False
            )
            self._write_test_summary(test_metrics)
            self._write_timing(timer.elapsed)
            self._write_alerts_file()
        return test_metrics

    def _run_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        global_step: int,
        total_steps: int,
    ) -> Dict[str, float]:
        self.model.train()
        losses: List[float] = []
        preds: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        energy_terms: List[torch.Tensor] = []

        # If using DistributedSampler, set epoch for shuffling
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            try:
                sampler.set_epoch(epoch)
            except Exception:
                pass
        progress = tqdm(
            loader, desc=f"epoch {epoch}", leave=False, disable=(self.rank != 0)
        )
        for batch in progress:
            if global_step >= total_steps:
                break
            batch = self._to_device(batch)
            step_start = time.perf_counter()
            if self.device.type == "cuda":
                amp_cm = torch.amp.autocast("cuda", enabled=bool(self.scaler))
            elif self.device.type == "mps" and hasattr(torch, "autocast"):
                amp_cm = torch.autocast(device_type="mps", enabled=self.config.use_amp)
            else:
                amp_cm = nullcontext()
            with amp_cm:
                outputs = self.model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("noise_level_ids"),
                )
                if not isinstance(outputs, ModelOutput):
                    if len(outputs) == 4:
                        logits, per_sample_energy, hidden_states, energy_components = (
                            outputs
                        )
                        if energy_components.numel() == 0:
                            energy_components = None
                    else:
                        logits, per_sample_energy, hidden_states = outputs
                        energy_components = None
                    outputs = ModelOutput(
                        logits=logits,
                        per_sample_energy=per_sample_energy,
                        hidden_states=hidden_states,
                        energy_components=energy_components,
                    )
                ce_loss = self.criterion(outputs.logits, batch["labels"])
            loss = ce_loss
            if self.energy_reg_weight > 0.0:
                energy_tensor = self._select_energy_for_regularization(outputs)
                if energy_tensor.numel() > 0:
                    reg = self._compute_energy_regularizer(energy_tensor)
                    loss = loss + self.energy_reg_weight * reg
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became non-finite.")
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimization.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimization.grad_clip
                )
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            global_step += 1
            self.step_times.append(time.perf_counter() - step_start)
            losses.append(loss.item())

            batch_preds = outputs.logits.argmax(dim=-1)
            preds.append(batch_preds.detach().cpu())
            labels.append(batch["labels"].detach().cpu())
            energy_terms.append(outputs.per_sample_energy.detach().cpu())

            progress.set_postfix({"loss": loss.item()})

        preds_tensor = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
        labels_tensor = (
            torch.cat(labels) if labels else torch.empty(0, dtype=torch.long)
        )
        preds_np = preds_tensor.numpy()
        labels_np = labels_tensor.numpy()
        acc = (
            metrics_lib.compute_accuracy(preds_np, labels_np) if preds_np.size else 0.0
        )
        macro_f1 = (
            metrics_lib.compute_macro_f1(preds_np, labels_np) if preds_np.size else 0.0
        )
        energy_tensor = (
            torch.cat(energy_terms)
            if energy_terms
            else torch.empty(0, dtype=torch.float32)
        )
        energy_array = energy_tensor.numpy()
        energy_mean = float(np.mean(energy_array)) if energy_array.size else 0.0
        energy_log_mean = (
            float(np.mean(np.log1p(np.maximum(energy_array, 0.0))))
            if energy_array.size
            else 0.0
        )
        ece = 0.0  # training split uses argmax only
        avg_loss = float(np.mean(losses)) if losses else 0.0
        if self.rank == 0:
            self.logger.info(
                "epoch=%s step=%s train_loss=%.4f acc=%.4f f1=%.4f energy=%.4f elog=%.4f",
                epoch,
                global_step,
                avg_loss,
                acc,
                macro_f1,
                energy_mean,
                energy_log_mean,
            )

        return {
            "loss": avg_loss,
            "acc": acc,
            "f1": macro_f1,
            "ece": ece,
            "energy": energy_mean,
            "energy_log": energy_log_mean,
            "energy_std": float(np.std(energy_array)) if energy_array.size else 0.0,
            "energy_p90": float(np.percentile(energy_array, 90))
            if energy_array.size
            else 0.0,
            "global_step": global_step,
        }

    def _select_energy_for_regularization(self, outputs: ModelOutput) -> torch.Tensor:
        energy = outputs.per_sample_energy
        if (
            self.config.energy_reg_scope == "last"
            and outputs.energy_components is not None
            and outputs.energy_components.numel() > 0
        ):
            energy = outputs.energy_components[-1]
        return energy

    def _compute_energy_regularizer(self, energy_tensor: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        clamped = energy_tensor.clamp_min(0.0)
        if self.config.energy_reg_mode == "absolute":
            return torch.log1p(clamped).mean()
        log_vals = torch.log1p(clamped + eps)
        centered = log_vals - log_vals.mean()
        return centered.pow(2).mean()

    def _maybe_backoff_energy_reg(self, energy_std: float, epoch: int) -> None:
        guard = getattr(self.config, "energy_guard", None)
        if guard is None or self.energy_reg_weight <= 0.0:
            return
        threshold = getattr(guard, "std_threshold", 0.0) or 0.0
        if threshold <= 0.0 or energy_std >= threshold:
            return
        min_weight = max(getattr(guard, "min_weight", 0.0), 0.0)
        if self.energy_reg_weight <= min_weight:
            return
        factor = getattr(guard, "factor", 0.5)
        if not (0.0 < factor < 1.0):
            factor = 0.5
        new_weight = max(min_weight, self.energy_reg_weight * factor)
        if new_weight >= self.energy_reg_weight:
            return
        prev = self.energy_reg_weight
        self.energy_reg_weight = new_weight
        if self.rank == 0:
            self.logger.warning(
                "energy_std=%.4f below guard %.4f -> lowering λ from %.2e to %.2e (epoch=%s)",
                energy_std,
                threshold,
                prev,
                new_weight,
                epoch,
            )
        self._record_alert(
            {
                "type": "guard_backoff",
                "epoch": epoch,
                "energy_std": energy_std,
                "threshold": threshold,
                "prev_weight": prev,
                "new_weight": new_weight,
            }
        )

    def _check_energy_alert(
        self, split: str, epoch: Optional[int], metrics: Dict[str, float]
    ) -> bool:
        watch = getattr(self.config, "energy_watch", None)
        if watch is None:
            return False
        triggered = False
        reasons: List[str] = []
        energy_std = metrics.get("energy_std", 0.0)
        std_threshold = getattr(watch, "std_threshold", None)
        if (
            std_threshold is not None
            and std_threshold > 0
            and energy_std < std_threshold
        ):
            triggered = True
            reasons.append(f"std<{std_threshold}")
        energy_p90 = metrics.get("energy_p90", 0.0)
        p90_threshold = getattr(watch, "p90_threshold", None)
        if (
            p90_threshold is not None
            and p90_threshold > 0
            and energy_p90 < p90_threshold
        ):
            triggered = True
            reasons.append(f"p90<{p90_threshold}")
        if triggered:
            self._record_alert(
                {
                    "type": "watch",
                    "split": split,
                    "epoch": epoch,
                    "energy_std": energy_std,
                    "energy_p90": energy_p90,
                    "reasons": reasons,
                }
            )
        return triggered

    def _record_alert(self, entry: Dict[str, Any]) -> None:
        if self.rank == 0:
            self.energy_alerts.append(entry)

    def _guard_config_dict(self) -> Dict[str, float]:
        guard = getattr(self.config, "energy_guard", None)
        if guard is None:
            return {}
        return {
            "std_threshold": getattr(guard, "std_threshold", 0.0) or 0.0,
            "factor": getattr(guard, "factor", 0.0),
            "min_weight": getattr(guard, "min_weight", 0.0),
        }

    def _watch_config_dict(self) -> Dict[str, float]:
        watch = getattr(self.config, "energy_watch", None)
        if watch is None:
            return {}
        payload: Dict[str, float] = {}
        if getattr(watch, "std_threshold", None):
            payload["std_threshold"] = getattr(watch, "std_threshold")
        if getattr(watch, "p90_threshold", None):
            payload["p90_threshold"] = getattr(watch, "p90_threshold")
        return payload

    def _write_alerts_file(self) -> None:
        if not self.energy_alerts:
            return
        payload = {
            "energy_guard": self._guard_config_dict(),
            "energy_watch": self._watch_config_dict(),
            "events": self.energy_alerts,
        }
        with open(self.save_dir / "alerts.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def evaluate(
        self,
        loader: DataLoader,
        split: str,
        epoch: Optional[int] = None,
        record_confusion: bool = False,
        compute_correlation: bool = False,
    ) -> Dict[str, float]:
        if self.rank != 0 and self.world_size > 1:
            # Skip validation/test on non-zero ranks to avoid duplication
            return {
                "loss": 0.0,
                "acc": 0.0,
                "macro_f1": 0.0,
                "ece": 0.0,
                "energy_mean": 0.0,
                "energy_log_mean": 0.0,
                "energy_std": 0.0,
                "energy_p90": 0.0,
            }
        self.model.eval()
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_energies: List[torch.Tensor] = []
        losses: List[float] = []

        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                if self.device.type == "cuda":
                    amp_cm = torch.amp.autocast("cuda", enabled=bool(self.scaler))
                elif self.device.type == "mps" and hasattr(torch, "autocast"):
                    amp_cm = torch.autocast(
                        device_type="mps", enabled=self.config.use_amp
                    )
                else:
                    amp_cm = nullcontext()
                with amp_cm:
                    outputs = self.model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch.get("noise_level_ids"),
                    )
                if not isinstance(outputs, ModelOutput):
                    if len(outputs) == 4:
                        logits, per_sample_energy, hidden_states, energy_components = (
                            outputs
                        )
                        if energy_components.numel() == 0:
                            energy_components = None
                    else:
                        logits, per_sample_energy, hidden_states = outputs
                        energy_components = None
                    outputs = ModelOutput(
                        logits=logits,
                        per_sample_energy=per_sample_energy,
                        hidden_states=hidden_states,
                        energy_components=energy_components,
                    )
                loss = self.criterion(outputs.logits, batch["labels"])
                losses.append(loss.item())
                all_logits.append(outputs.logits.detach().cpu())
                all_labels.append(batch["labels"].detach().cpu())
                all_energies.append(outputs.per_sample_energy.detach().cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        energies = torch.cat(all_energies, dim=0)
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = probs.argmax(axis=1)
        labels_np = labels.numpy()
        losses_np = np.array(losses) if losses else np.zeros(1)
        energies_np = energies.numpy()
        energy_log_mean = float(np.mean(np.log1p(np.maximum(energies_np, 0.0))))
        energy_std = float(np.std(energies_np)) if energies_np.size else 0.0
        energy_p90 = float(np.percentile(energies_np, 90)) if energies_np.size else 0.0
        metrics = {
            "loss": float(losses_np.mean()) if losses_np.size else 0.0,
            "acc": metrics_lib.compute_accuracy(preds, labels_np),
            "macro_f1": metrics_lib.compute_macro_f1(preds, labels_np),
            "ece": metrics_lib.expected_calibration_error(probs, labels_np),
            "energy_mean": float(energies.mean().item()),
            "energy_log_mean": energy_log_mean,
            "energy_std": energy_std,
            "energy_p90": energy_p90,
        }

        if record_confusion:
            matrix = metrics_lib.confusion_matrix(labels_np, preds, probs.shape[1])
            pd.DataFrame(matrix).to_csv(
                self.save_dir / "confusion_matrix.csv", index=False
            )

        if compute_correlation:
            errors = (preds != labels_np).astype(np.float32)
            energies_np = energies.numpy()
            if np.std(errors) == 0 or np.std(energies_np) == 0:
                pearson_r = 0.0
            else:
                pearson_r = float(np.corrcoef(errors, energies_np)[0, 1])
            payload = {
                "pearson_r": pearson_r,
                "n": int(len(errors)),
                "notes": "computed on test set",
            }
            with open(
                self.save_dir / "energy_error_correlation.json", "w", encoding="utf-8"
            ) as f:
                json.dump(payload, f, indent=2)
            if self.config.dump_energy_per_sample:
                df = pd.DataFrame(
                    {
                        "sample_idx": np.arange(len(energies_np)),
                        "energy": energies_np,
                        "error": errors,
                    }
                )
                df.to_csv(self.save_dir / "energy_per_sample.csv", index=False)

        tag = f"{split} epoch={epoch}" if epoch else split
        if self.rank == 0:
            self.logger.info(
                "%s loss=%.4f acc=%.4f f1=%.4f ece=%.4f energy=%.4f elog=%.4f",
                tag,
                metrics["loss"],
                metrics["acc"],
                metrics["macro_f1"],
                metrics["ece"],
                metrics["energy_mean"],
                metrics["energy_log_mean"],
            )
        return metrics

    def _write_test_summary(self, metrics: Dict[str, float]) -> None:
        summary = {
            "acc": metrics["acc"],
            "macro_f1": metrics["macro_f1"],
            "loss": metrics["loss"],
            "ece": metrics["ece"],
            "energy_mean_test": metrics["energy_mean"],
            "energy_log_mean_test": metrics["energy_log_mean"],
        }
        with open(self.save_dir / "test_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _write_timing(self, total_time: float) -> None:
        timing = {
            "total_train_sec": float(total_time),
            "avg_step_sec": float(np.mean(self.step_times)) if self.step_times else 0.0,
        }
        with open(self.save_dir / "timing.json", "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        non_blocking = self.device.type == "cuda"
        return {
            k: (v.to(self.device, non_blocking=non_blocking) if hasattr(v, "to") else v)
            for k, v in batch.items()
        }


def run_training(
    config: ExperimentConfig,
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],
    save_dir: Path,
) -> Dict[str, float]:
    """High-level convenience wrapper."""

    trainer = Trainer(
        model=model,
        config=config,
        save_dir=save_dir,
        device=torch.device(config.device),
    )
    return trainer.fit(loaders)
