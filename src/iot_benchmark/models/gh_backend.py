# -*- coding: utf-8 -*-
"""低容量、可审计的未来状态预测模型。

本模块只接受源端特征和训练折目标计数。目标端计数只在评分函数中出现，
因此可以在谱系分组外层验证中复用。M3 是本项目的门控 IOT 模型：

    p = alpha * q_iot + (1 - alpha) * r_supervised

状态组成使用 Dirichlet-multinomial 计数似然；可检测头使用源端特征的逻辑
回归。所有模型保持线性或低维，适合当前公开数据的样本量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, gammaln
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


EPS = 1e-9
LAMBDA_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def normalize_rows(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = np.clip(arr, 0.0, None)
    s = arr.sum(axis=1, keepdims=True)
    out = arr / np.maximum(s, eps)
    zero = s[:, 0] <= eps
    if np.any(zero):
        out[zero] = 1.0 / arr.shape[1]
    return out


def project_simplex(v: np.ndarray) -> np.ndarray:
    """欧氏投影到概率单纯形。"""
    x = np.asarray(v, dtype=float).ravel()
    if x.size == 0:
        return x.copy()
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, x.size + 1)
    ok = u - cssv / ind > 0
    if not np.any(ok):
        return np.full(x.size, 1.0 / x.size)
    rho = int(np.where(ok)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(x - theta, 0.0)


def project_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return project_simplex(arr)
    return np.vstack([project_simplex(row) for row in arr])


def cross_entropy(target_counts: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y = normalize_rows(target_counts)
    p = np.clip(normalize_rows(pred), EPS, 1.0)
    return -np.sum(y * np.log(p), axis=1)


def brier(target_counts: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y = normalize_rows(target_counts)
    p = normalize_rows(pred)
    return np.sum((p - y) ** 2, axis=1)


def total_variation(target_counts: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y = normalize_rows(target_counts)
    p = normalize_rows(pred)
    return 0.5 * np.abs(p - y).sum(axis=1)


def dirichlet_multinomial_logpmf(
    counts: np.ndarray, probs: np.ndarray, kappa: float
) -> np.ndarray:
    """逐行计算 Dirichlet-multinomial log PMF。"""
    y = np.asarray(counts, dtype=float)
    p = np.clip(normalize_rows(probs), EPS, 1.0)
    n = y.sum(axis=1)
    alpha = np.maximum(float(kappa) * p, EPS)
    result = gammaln(n + 1.0) - gammaln(y + 1.0).sum(axis=1)
    result += gammaln(float(kappa)) - gammaln(n + float(kappa))
    result += (gammaln(y + alpha) - gammaln(alpha)).sum(axis=1)
    return result


def dm_nll_per_cell(counts: np.ndarray, probs: np.ndarray, kappa: float) -> np.ndarray:
    n = np.asarray(counts, dtype=float).sum(axis=1)
    return -dirichlet_multinomial_logpmf(counts, probs, kappa) / np.maximum(n, 1.0)


def fit_kappa(counts: np.ndarray, probs: np.ndarray) -> float:
    """只使用训练折估计过度离散参数，设置稳定的有限边界。"""
    counts = np.asarray(counts, dtype=float)
    if counts.shape[0] < 4:
        return 20.0

    def objective(log_kappa: np.ndarray) -> float:
        kappa = float(np.exp(log_kappa[0]))
        return float(-np.sum(dirichlet_multinomial_logpmf(counts, probs, kappa)))

    fit = minimize(objective, np.array([np.log(20.0)]), method="L-BFGS-B", bounds=[(np.log(0.5), np.log(5000.0))])
    if not fit.success or not np.isfinite(fit.fun):
        return 20.0
    return float(np.exp(fit.x[0]))


def _augmented_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    return np.column_stack([np.ones(x.shape[0]), x])


@dataclass
class RidgePriorModel:
    prior_mode: str
    lam: float
    intercept_: Optional[np.ndarray] = None
    coef_: Optional[np.ndarray] = None
    k: Optional[int] = None

    def fit(
        self,
        x: np.ndarray,
        target_counts: np.ndarray,
        prior: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "RidgePriorModel":
        x_aug = _augmented_x(x)
        y = normalize_rows(target_counts)
        q = normalize_rows(prior)
        n, p = x_aug.shape
        if q.shape != y.shape:
            raise ValueError("prior 和 target_counts 维度不一致")
        w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float).ravel()
        w = np.clip(w, 0.1, None)
        gram = x_aug.T @ (w[:, None] * x_aug)
        rhs = x_aug.T @ (w[:, None] * (y - q))
        penalty = np.zeros((p, p), dtype=float)
        penalty[1:, 1:] = float(self.lam) * np.eye(p - 1)
        mat = gram + penalty
        try:
            delta = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(mat) @ rhs
        self.intercept_ = delta[0]
        self.coef_ = delta[1:]
        self.k = y.shape[1]
        return self

    def predict(self, x: np.ndarray, prior: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("模型尚未拟合")
        q = normalize_rows(prior)
        delta = _augmented_x(x) @ np.vstack([self.intercept_, self.coef_])
        return project_rows(q + delta)


def fit_ridge_prior(
    x: np.ndarray,
    target_counts: np.ndarray,
    prior: np.ndarray,
    lam: float,
    weights: Optional[np.ndarray] = None,
) -> RidgePriorModel:
    return RidgePriorModel(prior_mode="custom", lam=float(lam)).fit(x, target_counts, prior, weights)


def fit_direct_ridge(
    x: np.ndarray,
    target_counts: np.ndarray,
    lam: float,
    weights: Optional[np.ndarray] = None,
) -> RidgePriorModel:
    """监督基线，均匀分布作为中心，不使用 IOT。"""
    y = np.asarray(target_counts, dtype=float)
    uniform = np.full((y.shape[0], y.shape[1]), 1.0 / y.shape[1])
    return fit_ridge_prior(x, y, uniform, lam, weights)


def sinkhorn_plan(a: np.ndarray, b: np.ndarray, cost: np.ndarray, eps: float = 0.5, iters: int = 500) -> np.ndarray:
    """稳定的平衡熵正则 OT；a、b 只应来自训练折。"""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    c = np.asarray(cost, dtype=float)
    if c.shape != (a.size, b.size):
        raise ValueError("cost 形状与边际不一致")
    a = normalize_rows(a)[0] if a.ndim == 1 else normalize_rows(a)[0]
    b = normalize_rows(b)[0] if b.ndim == 1 else normalize_rows(b)[0]
    a = (a + 1e-6) / (a.sum() + 1e-6 * a.size)
    b = (b + 1e-6) / (b.sum() + 1e-6 * b.size)
    c = c - np.nanmin(c)
    scale = np.nanmedian(c[c > 0]) if np.any(c > 0) else 1.0
    c = c / max(float(scale), EPS)
    kernel = np.exp(-np.clip(c, 0.0, 60.0) / max(float(eps), 1e-4))
    kernel = np.maximum(kernel, 1e-300)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(int(iters)):
        new_u = a / np.maximum(kernel @ v, 1e-300)
        new_v = b / np.maximum(kernel.T @ new_u, 1e-300)
        if np.max(np.abs(new_u - u)) < 1e-10 and np.max(np.abs(new_v - v)) < 1e-10:
            u, v = new_u, new_v
            break
        u, v = new_u, new_v
    plan = u[:, None] * kernel * v[None, :]
    plan = plan / max(plan.sum(), EPS)
    return plan


def row_conditional(plan: np.ndarray) -> np.ndarray:
    p = np.asarray(plan, dtype=float)
    rows = p.sum(axis=1, keepdims=True)
    out = p / np.maximum(rows, EPS)
    zero = rows[:, 0] <= EPS
    if np.any(zero):
        out[zero] = 1.0 / p.shape[1]
    return out


def fit_iot_q(
    source_counts: np.ndarray,
    target_counts: np.ndarray,
    cost: np.ndarray,
    eps: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """由训练折的谱系聚合计数拟合 Q。"""
    xs = normalize_rows(source_counts)
    ys = normalize_rows(target_counts)
    weights = np.asarray(target_counts, dtype=float).sum(axis=1)
    weights = np.clip(weights, 1.0, None)
    m = np.einsum("ni,nj,n->ij", xs, ys, weights)
    a = np.asarray(source_counts, dtype=float).sum(axis=0)
    b = np.asarray(target_counts, dtype=float).sum(axis=0)
    if a.sum() <= 0 or b.sum() <= 0:
        k = source_counts.shape[1]
        return np.full((k, k), 1.0 / k), {"n_detected": 0, "fallback": True}
    plan = sinkhorn_plan(a, b, cost, eps=eps)
    q = row_conditional(plan)
    return q, {
        "n_detected": int(np.sum(weights > 0)),
        "source_mass": a.tolist(),
        "target_mass": b.tolist(),
        "plan": plan.tolist(),
        "transport_cost": float(np.sum(plan * cost)),
        "fallback": False,
        "empirical_joint": m.tolist(),
    }


def q_summaries(x: np.ndarray, q: np.ndarray, cost: np.ndarray) -> np.ndarray:
    q = normalize_rows(q)
    entropy = -np.sum(q * np.log(np.clip(q, EPS, 1.0)), axis=1)
    expected_cost = np.sum(q * cost[None, :, :].mean(axis=1), axis=1) if cost.ndim == 3 else np.sum(q * np.mean(cost, axis=1)[None, :], axis=1)
    offdiag = 1.0 - np.sum(q * np.eye(q.shape[1])[None, :, :].sum(axis=1), axis=1) if q.ndim == 3 else 1.0 - np.sum(q * np.eye(q.shape[1])[None, :, :], axis=(1, 2))
    # 每个谱系的源状态到目标状态的期望代价，使用源端状态组成加权。
    if q.ndim == 3:
        xq = np.einsum("ni,nij->nj", normalize_rows(x), q)
        expected_cost = np.sum(xq * np.mean(cost, axis=0)[None, :], axis=1)
        offdiag = 1.0 - np.sum(xq * np.eye(q.shape[1])[None, :, :].sum(axis=1), axis=1)
    return np.column_stack([entropy, expected_cost, offdiag])


def _gate_features(x: np.ndarray, q: np.ndarray, cost: np.ndarray) -> np.ndarray:
    q = normalize_rows(q)
    x = normalize_rows(x) if x.shape[1] == q.shape[1] else np.asarray(x, dtype=float)
    entropy = -np.sum(q * np.log(np.clip(q, EPS, 1.0)), axis=1, keepdims=True)
    # IOT 摘要只来自源端状态和训练折 Q。
    if q.shape[0] == q.shape[1] and x.shape[1] == q.shape[0]:
        xq = x @ q
    elif q.shape[0] == x.shape[0]:
        xq = q
    else:
        raise ValueError("q must be a shared square transition matrix or one prior row per case")
    state_cost = np.mean(cost, axis=1) if cost.ndim == 2 else np.mean(cost, axis=(1, 2))
    mean_cost = np.sum(xq * np.mean(cost, axis=0)[None, :], axis=1, keepdims=True)
    return np.column_stack([np.ones(x.shape[0]), x, q, entropy, mean_cost])


@dataclass
class GatedIOTModel:
    lam: float = 3.0
    gate_l2: float = 1.0
    loss_mode: str = "cross_entropy"
    gate_coef_: Optional[np.ndarray] = None
    supervised_: Optional[RidgePriorModel] = None
    kappa_: float = 20.0

    def fit(
        self,
        x: np.ndarray,
        source_counts: np.ndarray,
        target_counts: np.ndarray,
        q: np.ndarray,
        cost: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "GatedIOTModel":
        y = normalize_rows(target_counts)
        prior_uniform = np.full_like(y, 1.0 / y.shape[1])
        self.supervised_ = fit_ridge_prior(x, target_counts, prior_uniform, self.lam, weights)
        r = self.supervised_.predict(x, prior_uniform)
        q = normalize_rows(q)
        g = _gate_features(x, q, cost)
        w = np.ones(y.shape[0], dtype=float) if weights is None else np.asarray(weights, dtype=float).ravel()
        w = np.clip(w / max(np.mean(w), EPS), 0.1, 10.0)

        def objective(theta: np.ndarray) -> float:
            alpha = expit(g @ theta)[:, None]
            p = np.clip(alpha * q + (1.0 - alpha) * r, EPS, 1.0)
            if self.loss_mode == "square":
                loss = np.average(np.sum((p - y) ** 2, axis=1), weights=w)
            else:
                loss = np.average(-np.sum(y * np.log(p), axis=1), weights=w)
            # 0.5 作为中性 gate 的初始点，正则只作用于可识别的线性项。
            reg = 0.5 * self.gate_l2 * np.sum(theta[1:] ** 2) / max(g.shape[1], 1)
            return float(loss + reg)

        starts = [np.zeros(g.shape[1]), np.r_[[-1.0], np.zeros(g.shape[1] - 1)], np.r_[[1.0], np.zeros(g.shape[1] - 1)]]
        fits = [minimize(objective, s, method="L-BFGS-B", options={"maxiter": 500}) for s in starts]
        best = min((f for f in fits if np.isfinite(f.fun)), key=lambda f: f.fun, default=None)
        self.gate_coef_ = np.zeros(g.shape[1]) if best is None else np.asarray(best.x, dtype=float)
        self.kappa_ = fit_kappa(target_counts, y)
        return self

    def predict(self, x: np.ndarray, q: np.ndarray, cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.supervised_ is None or self.gate_coef_ is None:
            raise RuntimeError("门控模型尚未拟合")
        q = normalize_rows(q)
        uniform = np.full_like(q, 1.0 / q.shape[1])
        r = self.supervised_.predict(x, uniform)
        g = _gate_features(x, q, cost)
        alpha = expit(g @ self.gate_coef_)
        p = alpha[:, None] * q + (1.0 - alpha[:, None]) * r
        return normalize_rows(p), alpha


def fit_detection_model(
    x: np.ndarray,
    z: np.ndarray,
    c_values: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
) -> Tuple[object, float]:
    """拟合可检测头；单一类别时返回常数模型。"""
    y = np.asarray(z, dtype=int).ravel()
    if np.unique(y).size < 2:
        return ConstantProbability(float(np.mean(y))), 0.0
    best = None
    # 这里的 C 网格是预注册的；外层测试标签没有参与。
    for c in c_values:
        model = LogisticRegression(C=float(c), class_weight="balanced", solver="lbfgs", max_iter=1000)
        model.fit(x, y)
        prob = model.predict_proba(x)[:, 1]
        score = float(np.mean((prob - y) ** 2))
        if best is None or score < best[0]:
            best = (score, model, float(c))
    return best[1], best[2]


@dataclass
class ConstantProbability:
    value: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(np.asarray(x).shape[0], np.clip(self.value, EPS, 1.0 - EPS))
        return np.column_stack([1.0 - p, p])


def detection_metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int).ravel()
    p = np.clip(np.asarray(prob, dtype=float).ravel(), EPS, 1.0 - EPS)
    out = {
        "n": int(y.size),
        "positive_rate": float(np.mean(y)) if y.size else float("nan"),
        "auprc": float("nan"),
        "auroc": float("nan"),
        "brier": float(np.mean((p - y) ** 2)) if y.size else float("nan"),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if y.size else float("nan"),
    }
    if y.size and np.unique(y).size == 2:
        out["auprc"] = float(average_precision_score(y, p))
        out["auroc"] = float(roc_auc_score(y, p))
    return out


def state_metrics(counts: np.ndarray, pred: np.ndarray, kappa: float) -> Dict[str, float]:
    counts = np.asarray(counts, dtype=float)
    pred = normalize_rows(pred)
    ce = cross_entropy(counts, pred)
    br = brier(counts, pred)
    tv = total_variation(counts, pred)
    dm = dm_nll_per_cell(counts, pred, kappa)
    y = np.argmax(counts, axis=1)
    out = {
        "n_detected": int(counts.shape[0]),
        "cross_entropy": float(np.mean(ce)) if ce.size else float("nan"),
        "brier": float(np.mean(br)) if br.size else float("nan"),
        "total_variation": float(np.mean(tv)) if tv.size else float("nan"),
        "dm_nll_per_cell": float(np.mean(dm)) if dm.size else float("nan"),
        "argmax_accuracy": float(np.mean(np.argmax(pred, axis=1) == y)) if ce.size else float("nan"),
        "top2_accuracy": float(np.mean([y[i] in np.argsort(pred[i])[-2:] for i in range(len(y))])) if ce.size else float("nan"),
    }
    return out


def aggregate_metrics(records: Sequence[Mapping[str, object]], kappa: float) -> Dict[str, object]:
    """将逐记录预测聚合成报告字段。"""
    if not records:
        return {}
    y_z = np.asarray([int(r["target_detected"]) for r in records], dtype=int)
    p_z = np.asarray([float(r["p_detect"]) for r in records], dtype=float)
    target_counts = np.asarray([r["target_counts"] for r in records], dtype=float)
    state_pred = np.asarray([r["p_state"] for r in records], dtype=float)
    detect = detection_metrics(y_z, p_z)
    mask = y_z == 1
    state = state_metrics(target_counts[mask], state_pred[mask], kappa) if np.any(mask) else {}
    return {"detection": detect, "state": state, "n_records": int(len(records))}
