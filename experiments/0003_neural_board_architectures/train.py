"""Training loop: optax + equinox, wandb logging, best-val checkpointing.

One ``train`` call fits one architecture on one task and returns its held-out
metrics. Inputs are standardized on the train split (the engineered/global terms
span very different scales); the fitted stats are saved beside the checkpoint.
The win task is a win-probability logit (BCE, ranked by AUC); the heuristic task
is regression (RMSE / R^2).
"""

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from data import Dataset
from jaxtyping import Array, Float
from settlrl_learn.experiment import Run
from settlrl_learn.nn.graph import Sample
from sklearn.metrics import roc_auc_score


def _standardize(
    train: Dataset, val: Dataset
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    """Zero-mean/unit-var each feature on the train split; apply to both."""
    stats: dict[str, Any] = {}

    def fit_apply(field: str, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        x = getattr(train.samples, field)
        mean = x.mean(axis=axes, keepdims=True)
        std = np.maximum(x.std(axis=axes, keepdims=True), 1e-3)
        stats[field] = {"mean": mean.squeeze(0), "std": std.squeeze(0)}
        tr = (x - mean) / std
        v = (getattr(val.samples, field) - mean) / std
        return tr.astype(np.float32), v.astype(np.float32)

    tr_fields: dict[str, Any] = {}
    v_fields: dict[str, Any] = {}
    for field, axes in (
        ("nodes", (0, 1)), ("edges", (0, 1)), ("tiles", (0, 1)),
        ("glob", (0,)), ("extra", (0,)),
    ):  # fmt: skip
        tr_fields[field], v_fields[field] = fit_apply(field, axes)
    return (
        train._replace(samples=Sample(**tr_fields)),
        val._replace(samples=Sample(**v_fields)),
        stats,
    )


def _to_jax(samples: Sample) -> Sample:
    return cast(Sample, jax.tree.map(jnp.asarray, samples))


# Each task trains one or more heads. ``win`` is a binary outcome (BCE, ranked by
# AUC); the rest are regression (standardized MSE, ranked by R^2). ``multi`` shares
# one trunk across all four heads -- the supervised rehearsal for the AZ
# value+policy net, and a direct test of whether one trunk serves local
# (heuristic), global (win/turns) and structural (road) targets at once.
FIELD_KIND = {"win": "binary", "heur": "reg", "road": "reg", "turns": "reg"}
TASK_FIELDS: dict[str, list[str]] = {
    "win": ["win"],
    "heuristic": ["heur"],
    "road": ["road"],
    "turns": ["turns"],
    "multi": ["win", "heur", "road", "turns"],
}
_PRIMARY = {"win": "win", "heuristic": "heur", "road": "road", "turns": "turns",
            "multi": "win"}  # fmt: skip


def _suffix(field: str) -> str:
    return "auc" if FIELD_KIND[field] == "binary" else "r2"


def select_metric(task: str) -> str:
    """The held-out metric a task is ranked/gated by (its primary head)."""
    f = _PRIMARY[task]
    return f"{f}_{_suffix(f)}"


def _labels(
    train_ds: Dataset, val_ds: Dataset, fields: list[str]
) -> tuple[Float[Array, "n k"], np.ndarray, list[str]]:
    """Stack the task's fields into an ``(n, k)`` target matrix; standardize the
    regression columns on the train split so per-head MSE/BCE losses sum on a
    comparable scale (R^2 is scale-invariant, so metrics are unaffected)."""
    cols_tr, cols_val, kinds = [], [], []
    for f in fields:
        kind = FIELD_KIND[f]
        ytr = np.asarray(getattr(train_ds, f), np.float32)
        yval = np.asarray(getattr(val_ds, f), np.float32)
        if kind == "reg":
            mu, sd = ytr.mean(), max(float(ytr.std()), 1e-6)
            ytr, yval = (ytr - mu) / sd, (yval - mu) / sd
        cols_tr.append(ytr)
        cols_val.append(yval)
        kinds.append(kind)
    return jnp.asarray(np.stack(cols_tr, 1)), np.stack(cols_val, 1), kinds


def _metrics(
    fields: list[str], kinds: list[str], pred: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Per-head held-out metric (AUC for the binary head, R^2 for regression),
    keyed ``<field>_<auc|r2>``."""
    out: dict[str, float] = {}
    for k, (f, kind) in enumerate(zip(fields, kinds, strict=True)):
        p, t = pred[:, k], y[:, k]
        if kind == "binary":
            prob = 1.0 / (1.0 + np.exp(-p))
            out[f"{f}_auc"] = (
                float(roc_auc_score(t, prob)) if len(np.unique(t)) > 1 else float("nan")
            )
        else:
            sse = float(np.sum((p - t) ** 2))
            sst = float(np.sum((t - t.mean()) ** 2)) + 1e-9
            out[f"{f}_r2"] = 1.0 - sse / sst
    return out


def train(
    run: Run, cfg: dict, model: eqx.Module, train_ds: Dataset, val_ds: Dataset
) -> dict[str, float]:
    """Fit ``model`` (``out_dim`` = the task's head count) and return its best
    held-out per-head metrics, keyed ``best_<field>_<auc|r2>``. The model is
    selected/checkpointed on ``select_metric(task)`` (the primary head)."""
    task = cfg["task"]
    fields = TASK_FIELDS[task]
    train_ds, val_ds, stats = _standardize(train_ds, val_ds)
    run.save_json(
        "standardizer.json",
        {k: {m: a.tolist() for m, a in v.items()} for k, v in stats.items()},
    )

    x_tr, x_val = _to_jax(train_ds.samples), _to_jax(val_ds.samples)
    y_tr, y_val, kinds = _labels(train_ds, val_ds, fields)
    n = y_tr.shape[0]

    def loss_fn(m: Any, xs: Sample, y: Float[Array, "b k"]) -> Float[Array, ""]:
        pred = jax.vmap(m)(xs)  # (b, k)
        loss = jnp.asarray(0.0)
        for k, kind in enumerate(kinds):
            if kind == "binary":
                loss = (
                    loss
                    + optax.sigmoid_binary_cross_entropy(pred[:, k], y[:, k]).mean()
                )
            else:
                loss = loss + jnp.mean((pred[:, k] - y[:, k]) ** 2)
        return loss

    opt = optax.adamw(cfg["lr"], weight_decay=cfg["weight_decay"])
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(m: Any, st: Any, xs: Sample, y: Array) -> Any:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, xs, y)
        updates, st = opt.update(grads, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, updates), st, loss

    @eqx.filter_jit
    def predict(m: Any, xs: Sample) -> Array:
        return cast(Array, jax.vmap(m)(xs))  # (b, k)

    select = select_metric(task)
    best = -np.inf
    best_metrics: dict[str, float] = {}
    rng = np.random.default_rng(cfg["seed"])
    ckpt = run.dir / "best.eqx"
    bs = cfg["batch_size"]
    wb = wandb.init(
        project=cfg["wandb_project"], name=f"{cfg['arch']}-{task}",
        mode=cfg["wandb_mode"], config=cfg, reinit=True, dir=str(run.dir),
    )  # fmt: skip
    try:
        for epoch in range(cfg["epochs"]):
            order = rng.permutation(n)
            losses = []
            for i in range(0, n - bs + 1, bs):
                idx = jnp.asarray(order[i : i + bs])
                batch = jax.tree.map(lambda x: x[idx], x_tr)  # noqa: B023
                model, opt_state, loss = step(model, opt_state, batch, y_tr[idx])
                losses.append(float(loss))
            if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"] - 1:
                pred = np.asarray(predict(model, x_val))
                m = _metrics(fields, kinds, pred, y_val)
                train_loss = float(np.mean(losses)) if losses else float("nan")
                run.log(
                    epoch=epoch,
                    train_loss=train_loss,
                    **{f"val_{k}": v for k, v in m.items()},
                )
                wb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        **{f"val/{k}": v for k, v in m.items()},
                    }
                )
                if m[select] > best:
                    best = m[select]
                    best_metrics = m
                    eqx.tree_serialise_leaves(ckpt, model)
    finally:
        wb.finish()
    return {f"best_{k}": v for k, v in best_metrics.items()}
