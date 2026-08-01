"""Distillation trainer: the production net + production loss on a frozen
anchor-self-play dataset.

One ``distill_train`` call fits one ``GNNBackend``-built net with the backend's
own train step (masked policy CE + value logistic loss, production optimizer)
on minibatches from the frozen train dataset, and scores policy/value fit on
the independently generated val dataset. The value target is the production
blend ``(1-alpha)z + alpha*(q+1)/2`` applied at training time
(``steps.prepare_targets``), never at dump time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from omegaconf import OmegaConf
from settlrl_learn.experiment import Run
from settlrl_learn.nn.graph import Sample
from settlrl_learn.training.config import OptimConfig
from settlrl_learn.training.gnn_backend import GNNBackend, GNNItem
from settlrl_learn.training.steps import make_optimizer, prepare_targets

_ALPHAZERO_DIR = Path(__file__).resolve().parents[1] / "0004_alphazero"


def production_value_blend() -> float:
    """The production blend's alpha (``value_blend.max`` at full ramp), read
    from 0004's scale preset so the guard's value target can't drift from the
    loop's actual objective."""
    blend = OmegaConf.to_container(
        OmegaConf.load(_ALPHAZERO_DIR / "conf" / "value_blend" / "scale.yaml")
    )
    assert isinstance(blend, dict)
    return float(cast(Any, blend["max"]))


def production_optim() -> OptimConfig:
    """The production optimizer settings (lr / weight_decay / batch /
    grad_clip), read from 0004's optim scale preset so a production optim
    change can't silently de-sync the guard."""
    optim = OmegaConf.to_container(
        OmegaConf.load(_ALPHAZERO_DIR / "conf" / "optim" / "scale.yaml")
    )
    assert isinstance(optim, dict)
    return OptimConfig.model_validate(optim)


def _blend(data: dict[str, np.ndarray], alpha: float) -> np.ndarray:
    """The blended value target for ``data`` (the loop's own formula)."""
    blended, _ = prepare_targets(
        dict(data), blend=True, blend_max=alpha, blend_ramp=1, iteration=1
    )
    return np.asarray(blended["value"], np.float32)


@eqx.filter_jit
def _forward(net: Any, sample: Sample) -> tuple[Any, Any]:
    return cast(tuple[Any, Any], jax.vmap(net)(sample))


def _val_metrics(
    net: Any, item: GNNItem, blend_target: np.ndarray, chunk: int = 4096
) -> dict[str, float]:
    """Held-out fit: masked policy KL(recorded ‖ net), top-1 agreement, and the
    net's P(win) MSE vs the blended target and vs raw z (``item.value``)."""
    n = int(item.value.shape[0])
    vs_parts, logit_parts = [], []
    for i in range(0, n, chunk):
        part = cast(GNNItem, jax.tree.map(lambda x, i=i: x[i : i + chunk], item))
        vs, logits = _forward(net, Sample(part.nodes, part.edges, part.glob,
                                          part.tiles, None))  # fmt: skip
        vs_parts.append(np.asarray(vs))
        logit_parts.append(np.asarray(logits))
    vs_np = np.concatenate(vs_parts)
    logits_np = np.concatenate(logit_parts)

    legal = np.asarray(item.mask) > 0
    masked = np.where(legal, logits_np, -np.inf)
    peak = masked.max(axis=-1, keepdims=True)
    logp_net = masked - (
        peak + np.log(np.sum(np.exp(masked - peak), axis=-1, keepdims=True))
    )
    p_rec = np.asarray(item.policy)
    # logp_net is -inf only where p_rec is 0; zero it there so the discarded
    # `where` branch never forms 0 * inf.
    safe_logp = np.where(p_rec > 0, logp_net, 0.0)
    kl_terms = np.where(
        p_rec > 0, p_rec * (np.log(np.clip(p_rec, 1e-30, 1.0)) - safe_logp), 0.0
    )
    p_win = 1.0 / (1.0 + np.exp(-vs_np))
    z = np.asarray(item.value)
    return {
        "policy_kl": float(np.mean(np.sum(kl_terms, axis=-1))),
        "top1_agree": float(
            np.mean(np.argmax(masked, axis=-1) == np.argmax(p_rec, axis=-1))
        ),
        "value_mse_blend": float(np.mean((p_win - blend_target) ** 2)),
        "value_mse_z": float(np.mean((p_win - z) ** 2)),
    }


def distill_train(
    run: Run,
    cfg: dict,
    backend: GNNBackend,
    net: Any,
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
) -> dict[str, float]:
    """Fit ``net`` on ``train_data`` and return its val fit, keyed
    ``best_<metric>`` (checkpoint selection on policy KL, minimized) and
    ``final_<metric>`` (last epoch)."""
    alpha = production_value_blend()
    train_item = backend.to_item({**train_data, "value": _blend(train_data, alpha)})
    val_item = backend.to_item(dict(val_data))  # raw z rides in .value
    val_blend_target = _blend(val_data, alpha)

    optim = production_optim()
    optimizer = make_optimizer(optim)
    opt_state = backend.init_opt(optimizer, net)
    step = backend.make_step(optimizer)

    n = int(train_item.value.shape[0])
    bs = optim.batch_size
    rng = np.random.default_rng(cfg["seed"])
    best = np.inf
    best_metrics: dict[str, float] = {}
    final_metrics: dict[str, float] = {}
    ckpt = run.dir / "best.eqx"
    wb = wandb.init(
        project=cfg["wandb_project"], name=f"{cfg['arch']}-{cfg['task']}-s{cfg['seed']}",
        mode=cfg["wandb_mode"], config=cfg, reinit=True, dir=str(run.dir),
    )  # fmt: skip
    try:
        for epoch in range(cfg["epochs"]):
            order = rng.permutation(n)
            losses = []
            for i in range(0, n - bs + 1, bs):
                idx = jnp.asarray(order[i : i + bs])
                batch = cast(
                    GNNItem,
                    jax.tree.map(lambda x: x[idx], train_item),  # noqa: B023
                )
                net, opt_state, m = step(net, opt_state, batch)
                losses.append(float(m["loss"]))
            if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"] - 1:
                vm = _val_metrics(net, val_item, val_blend_target)
                final_metrics = vm
                train_loss = float(np.mean(losses)) if losses else float("nan")
                run.log(
                    epoch=epoch,
                    train_loss=train_loss,
                    **{f"val_{k}": v for k, v in vm.items()},
                )
                wb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        **{f"val/{k}": v for k, v in vm.items()},
                    }
                )
                if vm["policy_kl"] < best:
                    best = vm["policy_kl"]
                    best_metrics = vm
                    eqx.tree_serialise_leaves(ckpt, net)
    finally:
        wb.finish()
    return {
        **{f"best_{k}": v for k, v in best_metrics.items()},
        **{f"final_{k}": v for k, v in final_metrics.items()},
    }
