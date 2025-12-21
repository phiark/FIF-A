#!/usr/bin/env python3
"""
Phase 1 diagnostics: compare friction energy vs standard uncertainty signals.

Computes AUROC / Pearson-r / coverage-risk for:
- friction energy (optionally normalized per token)
- softmax entropy
- logit margin
- max-prob (inverted to uncertainty)

Usage:
    python scripts/diagnostics/compare_uncertainty.py \
        --run_dir result/1_0_4/sst2_noisy_low_hybrid_20251201_100304_seed42 \
        --checkpoint model.pt \
        --split test \
        --energy_norm per_token \
        --output phase1_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fif_mvp.cli.run_experiment import _build_data_bundle, _load_tokenizer_for_config
from fif_mvp.config import ExperimentConfig
from fif_mvp.models import ModelOutput, build_model
from fif_mvp.train import metrics as metrics_lib
from fif_mvp.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare friction energy vs softmax-based uncertainty."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Result directory containing config.json (and optionally checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (state_dict). Defaults to run_dir/model.pt if present.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--energy_norm",
        type=str,
        default="per_token",
        choices=["none", "per_token"],
        help="Optional normalization applied to energy before metrics.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size for evaluation (default: config.batch_size).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader workers for evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase1_results.json"),
        help="Path to write JSON summary.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> ExperimentConfig:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    raw = json.loads(cfg_path.read_text())
    return ExperimentConfig(**raw)


def build_eval_loader(
    config: ExperimentConfig, tokenizer, split: str, batch_size: int, num_workers: int
) -> DataLoader:
    # Reuse the data bundle builder to get the right preprocessing.
    # We only need one split; ignoring train noise settings for evaluation.
    bundle = _build_data_bundle(
        argparse.Namespace(
            workers=num_workers,
            task=config.task,
            sortish_batches=False,
            sortish_chunk_mult=50,
            noise_intensity=config.noise_intensity,
        ),
        config,
        tokenizer,
        config.train_noise_levels or config.noise_vocab,
    )
    loader = bundle.loaders.get(split)
    if loader is None:
        raise ValueError(f"Split '{split}' not available in data bundle.")
    # Override batch size & workers for eval stability
    loader = DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=loader.collate_fn,  # type: ignore[attr-defined]
    )
    return loader


def load_model(
    config: ExperimentConfig,
    tokenizer,
    checkpoint: Path | None,
    run_dir: Path,
    device: torch.device,
):
    model = build_model(
        config=config,
        vocab_size=config.vocab_size,
        num_labels=config.num_labels,
        pad_token_id=tokenizer.pad_token_id,
    )
    ckpt_path = checkpoint or (run_dir / "model.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        # Allow both direct state_dict and wrapped dict
        state_dict = (
            state.get("state_dict", state) if isinstance(state, dict) else state
        )
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(
            f"Checkpoint not found. Provide --checkpoint or place model.pt in {run_dir}"
        )
    model.to(device)
    model.eval()
    return model


def _energy_for_eval(
    outputs: ModelOutput, config: ExperimentConfig, attention_mask: torch.Tensor
) -> torch.Tensor:
    energy = outputs.per_sample_energy
    if (
        config.energy_reg_scope == "last"
        and outputs.energy_components is not None
        and outputs.energy_components.numel() > 0
    ):
        energy = outputs.energy_components[-1]
    # Optional per-token normalization
    return energy


@torch.no_grad()
def collect_scores(
    model,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    energy_norm: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    energies = []
    entropies = []
    margins = []
    max_probs = []
    labels = []
    logits_all = []
    for batch in loader:
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("noise_level_ids"),
        )
        if not isinstance(outputs, ModelOutput):
            # DataParallel tuple fallback
            if len(outputs) == 4:
                logits, per_sample_energy, hidden_states, energy_components = outputs
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
        probs = F.softmax(outputs.logits, dim=-1)
        entropies.append(-(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).cpu())
        top2 = outputs.logits.topk(2, dim=-1).values
        margins.append((top2[:, 0] - top2[:, 1]).cpu())
        max_probs.append(probs.max(dim=-1).values.cpu())
        labels.append(batch["labels"].cpu())
        logits_all.append(outputs.logits.detach().cpu())

        energy = _energy_for_eval(outputs, config, batch["attention_mask"])
        if energy_norm == "per_token":
            lengths = batch["attention_mask"].sum(dim=1).clamp_min(1)
            energy = energy / lengths
        energies.append(energy.detach().cpu())

    scores = {
        "energy": torch.cat(energies).numpy() if energies else np.array([]),
        "entropy": torch.cat(entropies).numpy() if entropies else np.array([]),
        "margin": torch.cat(margins).numpy() if margins else np.array([]),
        "max_prob": (1.0 - torch.cat(max_probs)).numpy() if max_probs else np.array([]),
    }
    labels_np = torch.cat(labels).numpy().astype(np.int64) if labels else np.array([])
    logits_np = torch.cat(logits_all).numpy() if logits_all else np.array([])
    return scores, labels_np, logits_np


def evaluate_uncertainties(
    scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    logits: np.ndarray,
    num_labels: int,
) -> Dict[str, Dict[str, float]]:
    preds = logits.argmax(axis=1)
    errors = (preds != labels).astype(np.int64)
    out: Dict[str, Dict[str, float]] = {}
    for name, arr in scores.items():
        if arr.size == 0:
            out[name] = {"auroc": 0.0, "pearson_r": 0.0, "aurc": 0.0}
            continue
        auroc = metrics_lib.safe_roc_auc(errors, arr)
        pearson = (
            float(np.corrcoef(arr, errors)[0, 1]) if np.unique(arr).size > 1 else 0.0
        )
        cov = metrics_lib.coverage_risk(errors, arr)
        out[name] = {
            "auroc": auroc,
            "pearson_r": pearson,
            "aurc": cov["aurc"],
            "risk_at_0.8": cov["risk_at"].get(0.8, 0.0),
            "risk_at_0.9": cov["risk_at"].get(0.9, 0.0),
            "risk_at_0.95": cov["risk_at"].get(0.95, 0.0),
        }
    return out


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(run_dir)
    set_seed(config.seed)
    tokenizer = _load_tokenizer_for_config(config, config.tokenizer_name)
    model = load_model(config, tokenizer, args.checkpoint, run_dir, device)
    loader = build_eval_loader(
        config=config,
        tokenizer=tokenizer,
        split=args.split,
        batch_size=args.batch_size or config.batch_size,
        num_workers=args.num_workers,
    )

    scores, labels_all, logits_all = collect_scores(
        model=model,
        loader=loader,
        config=config,
        device=device,
        energy_norm=args.energy_norm,
    )
    results = evaluate_uncertainties(scores, labels_all, logits_all, config.num_labels)

    # Ranking
    ranked = sorted(results.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
    for idx, (name, stats) in enumerate(ranked, 1):
        stats["rank"] = idx

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "split": args.split,
        "energy_norm": args.energy_norm,
        "device": str(device),
        "results": results,
        "ranked": ranked,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
