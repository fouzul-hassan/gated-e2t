"""
Lightweight hyperparameter sweep for decoding settings.

Focus metrics:
- BLEU@MTV (test/mean_BLEU1@MTV, test/mean_BLEU2@MTV)
- Retrieval accuracy (test/mean_retrieval_acc_top01/05/10)
- Zero-shot semantic classification (test/mean_*_cls_acc*)
- ETES (test/etes_alignment, test/etes_total, ...)

Example:
  python sweep_eval.py ^
    --checkpoint_path "./runs/glim-nucleus-gated-energy/checkpoints/epoch=epoch=009.ckpt" ^
    --data_path ./data/tmp/zuco_eeg_label_8variants.df ^
    --use_energy ^
    --gpus 0 ^
    --strategy_grid nucleus beam energy
"""

import argparse
from dataclasses import dataclass, asdict
from typing import Any

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from model.glim import GLIM
from data.datamodule import GLIMDataModule


@dataclass(frozen=True)
class DecodeConfig:
    generation_strategy: str
    num_beams: int | None = None
    top_p: float | None = None
    temperature: float | None = None
    energy_rerank_candidates: int | None = None


def _pick(metrics: dict[str, Any], key: str) -> float | None:
    v = metrics.get(key)
    if v is None:
        return None
    if hasattr(v, "item"):
        return float(v.item())
    try:
        return float(v)
    except Exception:
        return None


def _score(metrics: dict[str, Any]) -> float:
    """
    Simple scalar score for quick ranking.
    Higher is better.

    NOTE: ETES is "lower is better" (more negative is better alignment),
    so we subtract it (i.e., add -etes_alignment).
    """
    bleu1 = _pick(metrics, "test/mean_BLEU1@MTV") or 0.0
    ret1 = _pick(metrics, "test/mean_retrieval_acc_top01") or 0.0
    corpus = _pick(metrics, "test/mean_corpus_cls_acc") or 0.0
    etes = _pick(metrics, "test/etes_alignment")  # lower is better
    etes_term = (-etes) if etes is not None else 0.0
    # weights: prioritize BLEU@MTV and retrieval, lightly include zero-shot cls + ETES
    return 3.0 * bleu1 + 2.0 * ret1 + 1.0 * corpus + 0.5 * etes_term


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--use_energy", action="store_true", help="Enable ETES evaluation")
    p.add_argument("--gpus", default="0")
    p.add_argument("--bsz_test", type=int, default=24)
    p.add_argument("--limit_configs", type=int, default=0, help="If >0, only run first N configs")

    # Which strategy families to include
    p.add_argument("--strategy_grid", nargs="+", default=["beam", "nucleus"],
                   choices=["beam", "nucleus", "greedy", "energy"])

    args = p.parse_args()

    devices = [int(x) for x in args.gpus.split(",")]
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Data (shared)
    dm = GLIMDataModule(
        data_path=args.data_path,
        eval_noise_input=False,
        bsz_test=args.bsz_test,
        num_workers=4,
    )

    # Build a small grid
    configs: list[DecodeConfig] = []

    if "beam" in args.strategy_grid:
        for nb in [1, 2, 4]:
            configs.append(DecodeConfig("beam", num_beams=nb))

    if "nucleus" in args.strategy_grid:
        for top_p, temp in [(0.9, 0.7), (0.95, 0.7), (0.95, 1.0)]:
            configs.append(DecodeConfig("nucleus", top_p=top_p, temperature=temp))

    if "greedy" in args.strategy_grid:
        configs.append(DecodeConfig("greedy"))

    if "energy" in args.strategy_grid:
        for n_cand in [3, 5, 10]:
            configs.append(DecodeConfig("energy", energy_rerank_candidates=n_cand))

    if args.limit_configs and args.limit_configs > 0:
        configs = configs[: args.limit_configs]

    results: list[tuple[DecodeConfig, dict[str, Any]]] = []

    for cfg in configs:
        logger = WandbLogger(
            project="glim",
            name=f"eval-sweep-{cfg.generation_strategy}",
            save_dir="./runs/eval",
            offline=True,
        )

        overrides = {
            "generation_strategy": cfg.generation_strategy,
            "use_etes_eval": bool(args.use_energy),
            "use_energy_loss": False,
        }
        if cfg.num_beams is not None:
            overrides["num_beams"] = cfg.num_beams
        if cfg.top_p is not None:
            overrides["top_p"] = cfg.top_p
        if cfg.temperature is not None:
            overrides["temperature"] = cfg.temperature
        if cfg.energy_rerank_candidates is not None:
            overrides["energy_rerank_candidates"] = cfg.energy_rerank_candidates

        model = GLIM.load_from_checkpoint(
            args.checkpoint_path,
            map_location=f"cuda:{devices[0]}",
            strict=False,
            **overrides,
        )

        trainer = L.Trainer(
            accelerator="gpu",
            devices=devices,
            logger=logger,
            precision="bf16-mixed",
        )

        print("\n" + "=" * 80)
        print(f"CONFIG: {asdict(cfg)}")
        print("=" * 80)

        out = trainer.test(model, datamodule=dm)
        metrics = out[0] if out else {}
        results.append((cfg, metrics))

        print(
            "Quick score:",
            f"{_score(metrics):.4f}",
            "| BLEU1@MTV:",
            _pick(metrics, "test/mean_BLEU1@MTV"),
            "| Ret@1:",
            _pick(metrics, "test/mean_retrieval_acc_top01"),
            "| ETES:",
            _pick(metrics, "test/etes_alignment"),
        )

    # Rank and print summary
    ranked = sorted(results, key=lambda x: _score(x[1]), reverse=True)

    print("\n" + "=" * 80)
    print("SWEEP SUMMARY (sorted by quick score)")
    print("=" * 80)
    for cfg, m in ranked:
        print(
            f"{asdict(cfg)} | "
            f"score={_score(m):.4f} | "
            f"BLEU1@MTV={(_pick(m,'test/mean_BLEU1@MTV') or 0.0):.4f} | "
            f"BLEU2@MTV={(_pick(m,'test/mean_BLEU2@MTV') or 0.0):.4f} | "
            f"Ret@1={(_pick(m,'test/mean_retrieval_acc_top01') or 0.0):.4f} | "
            f"CorpusAcc={(_pick(m,'test/mean_corpus_cls_acc') or 0.0):.4f} | "
            f"RelTop1={(_pick(m,'test/mean_relation_cls_acc_top01') or 0.0):.4f} | "
            f"SentTop1={(_pick(m,'test/mean_sentiment_cls_acc_top01') or 0.0):.4f} | "
            f"ETES={_pick(m,'test/etes_alignment')}"
        )


if __name__ == "__main__":
    main()

