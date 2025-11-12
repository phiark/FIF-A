"""Training loop and evaluation pipelines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from contextlib import nullcontext
from torch import nn
import torch
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
            torch.amp.GradScaler("cuda") if (self.config.use_amp and device.type == "cuda") else None
        )
        self.rank = getattr(config, "rank", 0)
        self.world_size = getattr(config, "world_size", 1)

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
                train_metrics = self._run_epoch(
                    train_loader, epoch, global_step, total_steps
                )
                global_step = train_metrics["global_step"]
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
                    }
                )
                energy_rows.append(
                    {
                        "epoch": epoch,
                        "split": "train",
                        "energy_mean": train_metrics["energy"],
                        "energy_log_mean": train_metrics["energy_log"],
                    }
                )

                val_metrics = self.evaluate(
                    loaders["validation"], split="validation", epoch=epoch
                )
                metrics_rows.append(
                    {
                        "epoch": epoch,
                        "split": "validation",
                        **val_metrics,
                    }
                )
                energy_rows.append(
                    {
                        "epoch": epoch,
                        "split": "validation",
                        "energy_mean": val_metrics["energy_mean"],
                        "energy_log_mean": val_metrics["energy_log_mean"],
                    }
                )

                if global_step >= total_steps:
                    break

        test_metrics = self.evaluate(
            loaders["test"],
            split="test",
            record_confusion=True,
            compute_correlation=True,
        )
        metrics_rows.append({"epoch": 0, "split": "test", **test_metrics})
        energy_rows.append(
            {
                "epoch": 0,
                "split": "test",
                "energy_mean": test_metrics["energy_mean"],
                "energy_log_mean": test_metrics["energy_log_mean"],
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
        preds: List[int] = []
        labels: List[int] = []
        energy_terms: List[float] = []

        # If using DistributedSampler, set epoch for shuffling
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            try:
                sampler.set_epoch(epoch)
            except Exception:
                pass
        progress = tqdm(loader, desc=f"epoch {epoch}", leave=False, disable=(self.rank != 0))
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
                    logits, per_sample_energy, batch_energy, hidden_states = outputs
                    outputs = ModelOutput(
                        logits=logits,
                        per_sample_energy=per_sample_energy,
                        batch_energy=batch_energy,
                        hidden_states=hidden_states,
                    )
                ce_loss = self.criterion(outputs.logits, batch["labels"])
            loss = ce_loss
            if self.config.energy_reg_weight > 0.0:
                reg = torch.log1p(outputs.batch_energy.clamp_min(0.0))
                loss = loss + self.config.energy_reg_weight * reg
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
            preds.extend(batch_preds.detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())
            energy_terms.extend(outputs.per_sample_energy.detach().cpu().tolist())

            progress.set_postfix({"loss": loss.item()})

        preds_np = np.array(preds) if preds else np.zeros(1)
        labels_np = np.array(labels) if labels else np.zeros(1)
        acc = metrics_lib.compute_accuracy(preds_np, labels_np) if preds else 0.0
        macro_f1 = metrics_lib.compute_macro_f1(preds_np, labels_np) if preds else 0.0
        energy_array = np.array(energy_terms) if energy_terms else np.zeros(1)
        energy_mean = float(np.mean(energy_array))
        energy_log_mean = float(np.mean(np.log1p(np.maximum(energy_array, 0.0))))
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
            "global_step": global_step,
        }

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
                    logits, per_sample_energy, batch_energy, hidden_states = outputs
                    outputs = ModelOutput(
                        logits=logits,
                        per_sample_energy=per_sample_energy,
                        batch_energy=batch_energy,
                        hidden_states=hidden_states,
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
        metrics = {
            "loss": float(losses_np.mean()) if losses_np.size else 0.0,
            "acc": metrics_lib.compute_accuracy(preds, labels_np),
            "macro_f1": metrics_lib.compute_macro_f1(preds, labels_np),
            "ece": metrics_lib.expected_calibration_error(probs, labels_np),
            "energy_mean": float(energies.mean().item()),
            "energy_log_mean": energy_log_mean,
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
