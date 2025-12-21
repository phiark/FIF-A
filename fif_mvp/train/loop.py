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
from torch.nn import functional as F
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
        self.timing_steps = max(0, int(getattr(config, "timing_steps", 0) or 0))
        self.timing_warmup = max(0, int(getattr(config, "timing_warmup", 0) or 0))
        self.timing_records: List[Dict[str, float]] = []
        self._timing_seen = 0
        # AMP scaler for CUDA (support both torch.amp and torch.cuda.amp)
        if self.config.use_amp and device.type == "cuda":
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda")
            else:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
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
                        "energy_norm_mean": train_metrics["energy_norm_mean"],
                        "energy_norm_std": train_metrics["energy_norm_std"],
                        "energy_norm_p90": train_metrics["energy_norm_p90"],
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
                        "energy_norm_mean": train_metrics["energy_norm_mean"],
                        "energy_norm_std": train_metrics["energy_norm_std"],
                        "energy_norm_p90": train_metrics["energy_norm_p90"],
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
                        "energy_norm_mean": val_metrics.get("energy_norm_mean"),
                        "energy_norm_std": val_metrics.get("energy_norm_std"),
                        "energy_norm_p90": val_metrics.get("energy_norm_p90"),
                        "energy_alert": int(val_alert),
                        "energy_reg_weight": epoch_reg_weight,
                    }
                )

                self._maybe_adjust_energy_reg(train_metrics, epoch=epoch)

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
                "energy_norm_mean": test_metrics.get("energy_norm_mean"),
                "energy_norm_std": test_metrics.get("energy_norm_std"),
                "energy_norm_p90": test_metrics.get("energy_norm_p90"),
                "energy_alert": int(test_alert),
                "energy_reg_weight": self.energy_reg_weight,
                "energy_auroc": test_metrics.get("energy_auroc"),
                "energy_auprc": test_metrics.get("energy_auprc"),
                "coverage_aurc": test_metrics.get("coverage_aurc"),
                "coverage_risk_at_80": test_metrics.get("coverage_risk_at_80"),
                "coverage_risk_at_90": test_metrics.get("coverage_risk_at_90"),
                "coverage_risk_at_95": test_metrics.get("coverage_risk_at_95"),
                "energy_p90_correct": test_metrics.get("energy_p90_correct"),
                "energy_p90_incorrect": test_metrics.get("energy_p90_incorrect"),
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
            self._write_timing_breakdown()
            self._write_alerts_file()
            # Save model checkpoint
            if getattr(self.config, "save_model", False):
                self._save_checkpoint()
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
            except Exception as exc:
                if self.rank == 0:
                    self.logger.warning("Failed to set sampler epoch: %s", exc)
        progress = tqdm(
            loader, desc=f"epoch {epoch}", leave=False, disable=(self.rank != 0)
        )
        data_start = time.perf_counter()
        for batch in progress:
            if global_step >= total_steps:
                break
            record_timing = self._timing_should_record()
            batch = self._to_device(batch)
            if record_timing:
                self._maybe_sync()
                data_sec = time.perf_counter() - data_start
            else:
                data_sec = 0.0
            step_start = time.perf_counter()
            if self.device.type == "cuda":
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    amp_cm = torch.amp.autocast("cuda", enabled=bool(self.scaler))
                else:
                    amp_cm = torch.cuda.amp.autocast(enabled=bool(self.scaler))
            else:
                amp_cm = nullcontext()
            if record_timing:
                self._maybe_sync()
                fwd_start = time.perf_counter()
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
            if record_timing:
                self._maybe_sync()
                fwd_sec = time.perf_counter() - fwd_start
            loss = ce_loss
            if self.energy_reg_weight > 0.0:
                energy_tensor = self._select_energy_for_regularization(outputs)
                if energy_tensor.numel() > 0:
                    reg = self._compute_energy_regularizer(
                        energy_tensor, outputs.logits, batch["labels"]
                    )
                    loss = loss + self.energy_reg_weight * reg
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became non-finite.")
            if record_timing:
                self._maybe_sync()
                bwd_start = time.perf_counter()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimization.grad_clip
                )
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimization.grad_clip
                )
            if record_timing:
                self._maybe_sync()
                bwd_sec = time.perf_counter() - bwd_start
                self._maybe_sync()
                opt_start = time.perf_counter()
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            if record_timing:
                self._maybe_sync()
                opt_sec = time.perf_counter() - opt_start
            global_step += 1
            step_sec = time.perf_counter() - step_start
            self.step_times.append(step_sec)
            losses.append(loss.item())

            batch_preds = outputs.logits.argmax(dim=-1)
            preds.append(batch_preds.detach().cpu())
            labels.append(batch["labels"].detach().cpu())
            energy_eval = self._energy_for_eval(outputs)
            energy_terms.append(energy_eval.detach().cpu())

            progress.set_postfix({"loss": loss.item()})
            if record_timing:
                other_sec = step_sec - (fwd_sec + bwd_sec + opt_sec)
                self.timing_records.append(
                    {
                        "data_sec": float(data_sec),
                        "fwd_sec": float(fwd_sec),
                        "bwd_sec": float(bwd_sec),
                        "opt_sec": float(opt_sec),
                        "step_sec": float(step_sec),
                        "other_sec": float(max(other_sec, 0.0)),
                    }
                )
            data_start = time.perf_counter()

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
        energy_std = float(np.std(energy_array)) if energy_array.size else 0.0
        energy_norm = (
            (energy_array - energy_mean) / (energy_std + 1e-6)
            if energy_array.size
            else np.zeros_like(energy_array)
        )
        energy_norm_p90 = (
            float(np.percentile(energy_norm, 90)) if energy_norm.size else 0.0
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
            "energy_mean": energy_mean,
            "energy_log": energy_log_mean,
            "energy_std": energy_std,
            "energy_p90": float(np.percentile(energy_array, 90))
            if energy_array.size
            else 0.0,
            "energy_norm_mean": float(np.mean(energy_norm))
            if energy_norm.size
            else 0.0,
            "energy_norm_std": float(np.std(energy_norm)) if energy_norm.size else 0.0,
            "energy_norm_p90": energy_norm_p90,
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

    def _energy_for_eval(self, outputs: ModelOutput) -> torch.Tensor:
        """Energy tensor used for metrics/alerts, aligned with eval scope config."""

        energy = outputs.per_sample_energy
        if (
            self.config.energy_eval_scope == "auto"
            and self.config.energy_reg_scope == "last"
            and outputs.energy_components is not None
            and outputs.energy_components.numel() > 0
        ):
            energy = outputs.energy_components[-1]
        return energy

    def _compute_energy_regularizer(
        self,
        energy_tensor: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        eps = 1e-6
        clamped = energy_tensor.clamp_min(0.0)
        target = getattr(self.config, "energy_reg_target", None) or getattr(
            self.config, "energy_reg_mode", "absolute"
        )

        # Batch-normalize energy for ranking objectives to remove cross-task scale/shift.
        energy_mean = clamped.mean()
        energy_std = clamped.std(unbiased=False)
        energy_norm = (clamped - energy_mean) / (energy_std + eps)

        if target in {"margin", "rank"}:
            if logits is None or labels is None or energy_norm.numel() == 0:
                return torch.zeros(
                    (), device=energy_norm.device, dtype=energy_norm.dtype
                )
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct_mask = preds.eq(labels)
            correct_energy = energy_norm[correct_mask]
            wrong_energy = energy_norm[~correct_mask]
            if correct_energy.numel() == 0 or wrong_energy.numel() == 0:
                fallback = getattr(self.config, "energy_rank_fallback", "absolute")
                if fallback == "absolute":
                    return torch.log1p(clamped + eps).mean()
                return torch.zeros(
                    (), device=energy_norm.device, dtype=energy_norm.dtype
                )
            topk = getattr(self.config, "energy_rank_topk", 1) or 1
            topk = max(1, min(int(topk), wrong_energy.numel()))
            hardest_wrong = torch.topk(wrong_energy, k=topk, largest=True).values
            margin = float(getattr(self.config, "energy_rank_margin", 0.5))
            # Encourage correct energies to be lower than hardest wrong ones by at least margin.
            penalties = F.relu(
                margin + correct_energy.unsqueeze(1) - hardest_wrong.unsqueeze(0)
            )
            return penalties.mean()

        log_vals = torch.log1p(clamped + eps)
        if target == "normalized":
            centered = log_vals - log_vals.mean()
            return centered.pow(2).mean()

        # default absolute (scale control only)
        return log_vals.mean()

    def _maybe_adjust_energy_reg(self, metrics: Dict[str, float], epoch: int) -> None:
        guard = getattr(self.config, "energy_guard", None)
        if guard is None or self.energy_reg_weight <= 0.0:
            return
        target = getattr(self.config, "energy_reg_target", None) or getattr(
            self.config, "energy_reg_mode", "absolute"
        )
        std_val = metrics.get("energy_std", 0.0)
        p90_val = metrics.get("energy_p90", 0.0)
        factor_down = getattr(guard, "factor", 0.5)
        if not (0.0 < factor_down < 1.0):
            factor_down = 0.5
        factor_up = getattr(guard, "increase_factor", 1.0)
        if factor_up <= 1.0:
            factor_up = 1.0
        std_low = getattr(guard, "std_threshold", 0.0) or 0.0
        std_high = getattr(guard, "std_high_threshold", None)
        p90_low = getattr(guard, "p90_low_threshold", None)
        p90_high = getattr(guard, "p90_high_threshold", None)
        min_weight = max(getattr(guard, "min_weight", 0.0), 0.0)
        max_weight = getattr(guard, "max_weight", None)

        new_weight = self.energy_reg_weight
        reasons: List[Dict[str, Any]] = []

        if std_low > 0.0 and std_val < std_low and target not in {"margin", "rank"}:
            candidate = max(min_weight, new_weight * factor_down)
            if candidate < new_weight:
                reasons.append(
                    {
                        "metric": "std",
                        "direction": "low",
                        "value": float(std_val),
                        "threshold": float(std_low),
                    }
                )
                new_weight = candidate
        if p90_low is not None and p90_low > 0.0 and p90_val < p90_low:
            candidate = max(min_weight, new_weight * factor_down)
            if candidate < new_weight:
                reasons.append(
                    {
                        "metric": "p90",
                        "direction": "low",
                        "value": float(p90_val),
                        "threshold": float(p90_low),
                    }
                )
                new_weight = candidate

        if factor_up > 1.0:
            if std_high is not None and std_val > std_high:
                candidate = new_weight * factor_up
                if max_weight is not None:
                    candidate = min(max_weight, candidate)
                if candidate > new_weight:
                    reasons.append(
                        {
                            "metric": "std",
                            "direction": "high",
                            "value": float(std_val),
                            "threshold": float(std_high),
                        }
                    )
                    new_weight = candidate
            if p90_high is not None and p90_val > p90_high:
                candidate = new_weight * factor_up
                if max_weight is not None:
                    candidate = min(max_weight, candidate)
                if candidate > new_weight:
                    reasons.append(
                        {
                            "metric": "p90",
                            "direction": "high",
                            "value": float(p90_val),
                            "threshold": float(p90_high),
                        }
                    )
                    new_weight = candidate

        if max_weight is not None:
            new_weight = min(max_weight, new_weight)

        if not reasons or new_weight == self.energy_reg_weight:
            return

        prev = self.energy_reg_weight
        self.energy_reg_weight = new_weight
        if self.rank == 0:
            direction = "adjusting"
            self.logger.warning(
                "guard %s -> λ: %.2e -> %.2e (epoch=%s, reasons=%s)",
                direction,
                prev,
                new_weight,
                epoch,
                ";".join(f"{r['metric']}({r['direction']})" for r in reasons),
            )
        self._record_alert(
            {
                "type": "guard_adjust",
                "epoch": epoch,
                "prev_weight": prev,
                "new_weight": new_weight,
                "reasons": reasons,
                "min_weight": min_weight,
                "max_weight": max_weight,
                "factor_down": factor_down,
                "factor_up": factor_up,
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
        std_high = getattr(watch, "std_high_threshold", None)
        if std_high is not None and std_high > 0 and energy_std > std_high:
            triggered = True
            reasons.append(f"std>{std_high}")
        energy_p90 = metrics.get("energy_p90", 0.0)
        p90_threshold = getattr(watch, "p90_threshold", None)
        if (
            p90_threshold is not None
            and p90_threshold > 0
            and energy_p90 < p90_threshold
        ):
            triggered = True
            reasons.append(f"p90<{p90_threshold}")
        p90_high = getattr(watch, "p90_high_threshold", None)
        if p90_high is not None and p90_high > 0 and energy_p90 > p90_high:
            triggered = True
            reasons.append(f"p90>{p90_high}")
        energy_mean = metrics.get("energy_mean", metrics.get("energy", 0.0))
        mean_low = getattr(watch, "mean_low_threshold", None)
        if mean_low is not None and mean_low > 0 and energy_mean < mean_low:
            triggered = True
            reasons.append(f"mean<{mean_low}")
        mean_high = getattr(watch, "mean_high_threshold", None)
        if mean_high is not None and mean_high > 0 and energy_mean > mean_high:
            triggered = True
            reasons.append(f"mean>{mean_high}")
        if triggered:
            self._record_alert(
                {
                    "type": "watch",
                    "split": split,
                    "epoch": epoch,
                    "energy_std": energy_std,
                    "energy_p90": energy_p90,
                    "energy_mean": energy_mean,
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
        payload = {
            "std_threshold": getattr(guard, "std_threshold", 0.0) or 0.0,
            "factor": getattr(guard, "factor", 0.0),
            "min_weight": getattr(guard, "min_weight", 0.0),
        }
        if getattr(guard, "std_high_threshold", None) is not None:
            payload["std_high_threshold"] = getattr(guard, "std_high_threshold")
        if getattr(guard, "p90_low_threshold", None) is not None:
            payload["p90_low_threshold"] = getattr(guard, "p90_low_threshold")
        if getattr(guard, "p90_high_threshold", None) is not None:
            payload["p90_high_threshold"] = getattr(guard, "p90_high_threshold")
        if getattr(guard, "max_weight", None) is not None:
            payload["max_weight"] = getattr(guard, "max_weight")
        if getattr(guard, "increase_factor", None) is not None:
            payload["increase_factor"] = getattr(guard, "increase_factor")
        return payload

    def _watch_config_dict(self) -> Dict[str, float]:
        watch = getattr(self.config, "energy_watch", None)
        if watch is None:
            return {}
        payload: Dict[str, float] = {}
        if getattr(watch, "std_threshold", None):
            payload["std_threshold"] = getattr(watch, "std_threshold")
        if getattr(watch, "std_high_threshold", None):
            payload["std_high_threshold"] = getattr(watch, "std_high_threshold")
        if getattr(watch, "p90_threshold", None):
            payload["p90_threshold"] = getattr(watch, "p90_threshold")
        if getattr(watch, "p90_high_threshold", None):
            payload["p90_high_threshold"] = getattr(watch, "p90_high_threshold")
        if getattr(watch, "mean_low_threshold", None):
            payload["mean_low_threshold"] = getattr(watch, "mean_low_threshold")
        if getattr(watch, "mean_high_threshold", None):
            payload["mean_high_threshold"] = getattr(watch, "mean_high_threshold")
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
                "energy_norm_mean": 0.0,
                "energy_norm_std": 0.0,
                "energy_norm_p90": 0.0,
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
                    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                        amp_cm = torch.amp.autocast("cuda", enabled=bool(self.scaler))
                    else:
                        amp_cm = torch.cuda.amp.autocast(enabled=bool(self.scaler))
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
                energy_eval = self._energy_for_eval(outputs)
                all_energies.append(energy_eval.detach().cpu())

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
        energy_norm = (
            (energies_np - energies_np.mean()) / (energy_std + 1e-6)
            if energies_np.size
            else energies_np
        )
        energy_norm_std = float(np.std(energy_norm)) if energy_norm.size else 0.0
        energy_norm_p90 = (
            float(np.percentile(energy_norm, 90)) if energy_norm.size else 0.0
        )
        metrics = {
            "loss": float(losses_np.mean()) if losses_np.size else 0.0,
            "acc": metrics_lib.compute_accuracy(preds, labels_np),
            "macro_f1": metrics_lib.compute_macro_f1(preds, labels_np),
            "ece": metrics_lib.expected_calibration_error(probs, labels_np),
            "energy_mean": float(energies.mean().item()),
            "energy_log_mean": energy_log_mean,
            "energy_std": energy_std,
            "energy_p90": energy_p90,
            "energy_norm_mean": float(energy_norm.mean()) if energy_norm.size else 0.0,
            "energy_norm_std": energy_norm_std,
            "energy_norm_p90": energy_norm_p90,
        }

        if record_confusion:
            matrix = metrics_lib.confusion_matrix(labels_np, preds, probs.shape[1])
            pd.DataFrame(matrix).to_csv(
                self.save_dir / "confusion_matrix.csv", index=False
            )

        if compute_correlation:
            errors = (preds != labels_np).astype(np.float32)
            corr_energy = (
                energy_norm
                if getattr(self.config, "energy_metrics_source", "normalized")
                == "normalized"
                else energies_np
            )
            if np.std(errors) == 0 or np.std(corr_energy) == 0:
                pearson_r = 0.0
            else:
                pearson_r = float(np.corrcoef(errors, corr_energy)[0, 1])
            auroc = metrics_lib.safe_roc_auc(errors, corr_energy)
            auprc = metrics_lib.safe_average_precision(errors, corr_energy)
            coverage = metrics_lib.coverage_risk(errors, corr_energy)
            quantiles = metrics_lib.split_energy_quantiles(errors, corr_energy)
            coverage_curve = []
            cov_arr = coverage.get("coverages")
            risk_arr = coverage.get("risks")
            if cov_arr is not None and risk_arr is not None and len(cov_arr) > 0:
                cov_arr = coverage["coverages"]
                risk_arr = coverage["risks"]
                max_points = 200
                if cov_arr.shape[0] <= max_points:
                    idx = range(cov_arr.shape[0])
                else:
                    idx = np.linspace(
                        0, cov_arr.shape[0] - 1, num=max_points, dtype=int
                    )
                coverage_curve = [
                    {"coverage": float(cov_arr[i]), "risk": float(risk_arr[i])}
                    for i in idx
                ]
            metrics.update(
                {
                    "energy_auroc": auroc,
                    "energy_auprc": auprc,
                    "coverage_aurc": coverage.get("aurc", 0.0),
                    "coverage_risk_at_80": coverage.get("risk_at", {}).get(0.8, 0.0),
                    "coverage_risk_at_90": coverage.get("risk_at", {}).get(0.9, 0.0),
                    "coverage_risk_at_95": coverage.get("risk_at", {}).get(0.95, 0.0),
                    "energy_p50_correct": quantiles["correct"]["p50"],
                    "energy_p90_correct": quantiles["correct"]["p90"],
                    "energy_p99_correct": quantiles["correct"]["p99"],
                    "energy_p50_incorrect": quantiles["incorrect"]["p50"],
                    "energy_p90_incorrect": quantiles["incorrect"]["p90"],
                    "energy_p99_incorrect": quantiles["incorrect"]["p99"],
                }
            )
            payload = {
                "pearson_r": pearson_r,
                "auroc": auroc,
                "auprc": auprc,
                "aurc": coverage.get("aurc", 0.0),
                "coverage_risk_at": {
                    str(k): float(v) for k, v in coverage.get("risk_at", {}).items()
                },
                "coverage_curve": coverage_curve,
                "quantiles": quantiles,
                "n": int(len(errors)),
                "energy_metrics_source": getattr(
                    self.config, "energy_metrics_source", "normalized"
                ),
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
                        "energy": corr_energy,
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
        optional_keys = [
            "energy_auroc",
            "energy_auprc",
            "coverage_aurc",
            "coverage_risk_at_80",
            "coverage_risk_at_90",
            "coverage_risk_at_95",
            "energy_p90_correct",
            "energy_p90_incorrect",
            "energy_norm_mean",
            "energy_norm_std",
            "energy_norm_p90",
        ]
        for key in optional_keys:
            if key in metrics:
                summary[key] = metrics[key]
        with open(self.save_dir / "test_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _write_timing(self, total_time: float) -> None:
        timing = {
            "total_train_sec": float(total_time),
            "avg_step_sec": float(np.mean(self.step_times)) if self.step_times else 0.0,
        }
        with open(self.save_dir / "timing.json", "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)

    def _write_timing_breakdown(self) -> None:
        if not self.timing_records:
            return
        segments = {
            "data_sec",
            "fwd_sec",
            "bwd_sec",
            "opt_sec",
            "other_sec",
            "step_sec",
        }
        stats = {}
        for key in segments:
            values = [row[key] for row in self.timing_records]
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p90": float(np.percentile(values, 90)),
            }
        payload = {
            "timing_steps": int(self.timing_steps),
            "timing_warmup": int(self.timing_warmup),
            "samples": len(self.timing_records),
            "stats": stats,
            "steps": self.timing_records,
            "note": "segment timings use torch.cuda.synchronize when enabled",
        }
        with open(self.save_dir / "timing_breakdown.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _maybe_sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _timing_should_record(self) -> bool:
        if self.timing_steps <= 0:
            return False
        should_record = (
            self._timing_seen >= self.timing_warmup
            and len(self.timing_records) < self.timing_steps
        )
        self._timing_seen += 1
        return should_record

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        non_blocking = self.device.type == "cuda"
        return {
            k: (v.to(self.device, non_blocking=non_blocking) if hasattr(v, "to") else v)
            for k, v in batch.items()
        }

    def _save_checkpoint(self) -> None:
        """Save model checkpoint to save_dir/model.pt"""
        checkpoint_path = self.save_dir / "model.pt"
        try:
            # Get underlying model (unwrap if using DataParallel/DistributedDataParallel)
            model_to_save = self.model
            if hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module

            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else None,
            }, checkpoint_path)

            if self.rank == 0:
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as exc:
            if self.rank == 0:
                self.logger.error(f"Failed to save checkpoint: {exc}")


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
