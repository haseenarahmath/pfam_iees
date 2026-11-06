
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.iees_utils import (
    DEFAULT_COMMON_RESOLUTION,
    calculate_consistency_index,
    compute_attribution_strength,
    generate_pfams,
    minmax_normalize,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ProxyFeaturePack:
    X: np.ndarray
    y_iees: np.ndarray
    y_correct: np.ndarray
    y_conf: np.ndarray
    exits: np.ndarray


@dataclass
class ProxyModels:
    rf_iees: RandomForestRegressor
    rf_correct: RandomForestClassifier
    rf_conf: RandomForestRegressor


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_proxy_features(save_path: str, pack: ProxyFeaturePack) -> None:
    """Save proxy features and all supervision targets."""
    dirname = os.path.dirname(save_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    np.savez(
        save_path,
        X=pack.X,
        y_iees=pack.y_iees,
        y_correct=pack.y_correct,
        y_conf=pack.y_conf,
        exits=pack.exits,
    )
    print(f"[Cache] Features saved -> {save_path}")


def load_proxy_features(save_path: str) -> ProxyFeaturePack:
    """Load proxy features and all supervision targets."""
    data = np.load(save_path)
    print(f"[Cache] Features loaded <- {save_path}")
    return ProxyFeaturePack(
        X=data["X"],
        y_iees=data["y_iees"],
        y_correct=data["y_correct"],
        y_conf=data["y_conf"],
        exits=data["exits"],
    )


# ---------------------------------------------------------------------------
# 10-feature extraction
# ---------------------------------------------------------------------------

def _safe_num_exits(model) -> int:
    if not hasattr(model, "num_exits"):
        raise AttributeError("Model must expose model.num_exits")
    return int(model.num_exits)


def _feature_from_logits_and_activation(
    logits: torch.Tensor,
    activations: Optional[torch.Tensor],
    prev_state: Optional[dict],
) -> Tuple[np.ndarray, dict, float, int, float]:
    """
    Build the paper's 10 attribution-free proxy features for one sample/exit.

    Returns
    -------
    feature : np.ndarray shape (10,)
    state   : dict for next exit
    conf    : float
    pred    : int
    entropy : float
    """
    probs = F.softmax(logits, dim=1)
    conf = float(probs.max(dim=1).values.item())
    pred = int(probs.argmax(dim=1).item())

    top2 = torch.topk(probs, 2, dim=1).values
    margin = float((top2[:, 0] - top2[:, 1]).item())
    entropy = float((-(probs * torch.log(probs + 1e-8))).sum(dim=1).item())

    if activations is not None:
        act_mean = float(activations.abs().mean().item())
        act_max = float(activations.abs().amax().item())
    else:
        # Fallback for models where activations are not exposed during runtime.
        # Still keeps the feature vector well-defined.
        act_mean = float(logits.abs().mean().item())
        act_max = float(logits.abs().amax().item())

    if prev_state is None:
        conf_delta = 0.0
        class_stability = 0.0
        logit_cos = 0.0
        entropy_delta = 0.0
        act_delta = 0.0
    else:
        conf_delta = abs(conf - float(prev_state["conf"]))
        class_stability = float(pred == int(prev_state["pred"]))
        logit_cos = float(
            F.cosine_similarity(logits.detach(), prev_state["logits"], dim=1).item()
        )
        entropy_delta = abs(entropy - float(prev_state["entropy"]))
        act_delta = abs(act_mean - float(prev_state["act_mean"]))

    feature = np.array(
        [
            conf,
            margin,
            entropy,
            act_mean,
            act_max,
            conf_delta,
            class_stability,
            logit_cos,
            entropy_delta,
            act_delta,
        ],
        dtype=np.float32,
    )

    state = {
        "conf": conf,
        "pred": pred,
        "logits": logits.detach(),
        "entropy": entropy,
        "act_mean": act_mean,
    }
    return feature, state, conf, pred, entropy


def extract_proxy_features_10_oracle_iees(
    model,
    dataloader,
    device: torch.device | str,
    weights: Sequence[float],
    k: int = 20,
    common_resolution: Tuple[int, int] = DEFAULT_COMMON_RESOLUTION,
    desc: str = "Extracting proxy features",
) -> ProxyFeaturePack:
    """
    Extract 10-feature attribution-free inputs and three supervision targets.

    Oracle targets are computed offline:
      - Grad-CAM/PFAM maps are generated using model activations/gradients.
      - F_i is top-k% raw channel-weighted attribution strength.
      - Q_i is mean absolute change between cumulative PFAM maps.
      - F and Q are min-max normalized across this extraction split.
      - IEES = w1*S + w2*F_norm - w3*Q_norm.

    Runtime proxy inference should NOT call this function.
    """
    device = torch.device(device)
    model.eval()
    num_exits = _safe_num_exits(model)
    w1, w2, w3 = float(weights[0]), float(weights[1]), float(weights[2])

    features: List[np.ndarray] = []
    S_values: List[float] = []
    F_values: List[float] = []
    Q_values: List[float] = []
    y_correct: List[int] = []
    y_conf: List[float] = []
    exits: List[int] = []

    print(desc)
    print("  Offline oracle IEES extraction: PFAM/Grad-CAM enabled")
    print("  Runtime comparison will use only the saved 10 features")

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx % 25 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}")

        inputs = inputs.to(device)
        labels = labels.to(device)

        for sample_idx in range(inputs.size(0)):
            x = inputs[sample_idx : sample_idx + 1]
            y = int(labels[sample_idx].item())

            pfams_list: List[np.ndarray] = []
            cumulative_maps: List[np.ndarray] = []
            channel_weighted_maps: List[np.ndarray] = []
            prev_state: Optional[dict] = None

            for exit_idx in range(num_exits):
                model.zero_grad(set_to_none=True)

                logits, class_idx, _target_score = model.forward_to_exit(
                    x, exit_idx, True
                )

                # generate_pfams returns PFAM map, cumulative map, and raw
                # channel-weighted map list in the final utilities.
                pfams_list, cumulative_map, channel_weighted_maps = generate_pfams(
                    model.activations,
                    model.gradients,
                    x,
                    pfams_list,
                    channel_weighted_maps,
                    common_resolution=common_resolution,
                )
                cumulative_maps.append(cumulative_map)

                Q_raw = calculate_consistency_index(cumulative_maps)
                F_raw = compute_attribution_strength(
                    channel_weighted_maps[-1], k=k
                )

                feature, prev_state, conf, pred, _entropy = _feature_from_logits_and_activation(
                    logits, model.activations, prev_state
                )

                features.append(feature)
                S_values.append(conf)
                F_values.append(float(F_raw))
                Q_values.append(float(Q_raw))
                y_correct.append(int(pred == y))
                y_conf.append(float(conf))
                exits.append(exit_idx)

                # Avoid stale activations/gradients across exits/samples.
                model.activations = None
                model.gradients = None

    X = np.asarray(features, dtype=np.float32)
    S_arr = np.asarray(S_values, dtype=np.float64)
    F_arr = np.asarray(F_values, dtype=np.float64)
    Q_arr = np.asarray(Q_values, dtype=np.float64)

    F_norm = minmax_normalize(F_arr)
    Q_norm = minmax_normalize(Q_arr)
    y_iees = (w1 * S_arr + w2 * F_norm - w3 * Q_norm).astype(np.float32)

    pack = ProxyFeaturePack(
        X=X,
        y_iees=y_iees,
        y_correct=np.asarray(y_correct, dtype=np.int32),
        y_conf=np.asarray(y_conf, dtype=np.float32),
        exits=np.asarray(exits, dtype=np.int32),
    )

    print(f"  Done: X={pack.X.shape}, exits={num_exits}")
    print(
        f"  IEES range: {pack.y_iees.min():.3f} to {pack.y_iees.max():.3f} | "
        f"Correctness rate: {pack.y_correct.mean():.3f}"
    )
    return pack


# ---------------------------------------------------------------------------
# Proxy training and fidelity
# ---------------------------------------------------------------------------

def train_proxy_supervision_baselines(
    train_pack: ProxyFeaturePack,
    random_state: int = 42,
) -> ProxyModels:
    """Train IEES RF, correctness RF, and confidence-recalibration RF."""
    X = _ensure_2d(train_pack.X)

    rf_iees = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    rf_iees.fit(X, train_pack.y_iees)

    rf_correct = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    rf_correct.fit(X, train_pack.y_correct.astype(int))

    rf_conf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    rf_conf.fit(X, train_pack.y_conf)

    return ProxyModels(rf_iees=rf_iees, rf_correct=rf_correct, rf_conf=rf_conf)


def report_proxy_fidelity(models: ProxyModels, eval_pack: ProxyFeaturePack) -> Dict[str, Dict[str, float]]:
    """Report fidelity metrics for the three proxy supervision targets."""
    X = _ensure_2d(eval_pack.X)
    metrics: Dict[str, Dict[str, float]] = {}

    pred_iees = models.rf_iees.predict(X)
    metrics["IEES RF"] = {
        "r2": float(r2_score(eval_pack.y_iees, pred_iees)),
        "mae": float(mean_absolute_error(eval_pack.y_iees, pred_iees)),
        "rmse": float(np.sqrt(mean_squared_error(eval_pack.y_iees, pred_iees))),
    }

    pred_conf = models.rf_conf.predict(X)
    metrics["Conf. Recalib RF"] = {
        "r2": float(r2_score(eval_pack.y_conf, pred_conf)),
        "mae": float(mean_absolute_error(eval_pack.y_conf, pred_conf)),
        "rmse": float(np.sqrt(mean_squared_error(eval_pack.y_conf, pred_conf))),
    }

    prob_correct = models.rf_correct.predict_proba(X)[:, 1]
    pred_correct = (prob_correct >= 0.5).astype(int)
    metrics["Correctness RF"] = {
        "accuracy": float(accuracy_score(eval_pack.y_correct, pred_correct)),
        "auc": float(roc_auc_score(eval_pack.y_correct, prob_correct))
        if len(np.unique(eval_pack.y_correct)) > 1
        else float("nan"),
    }

    print("\nProxy fidelity")
    for name, vals in metrics.items():
        print(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in vals.items()))
    return metrics


def save_proxy_models(models: ProxyModels, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(models.rf_iees, os.path.join(save_dir, "rf_iees.joblib"))
    joblib.dump(models.rf_correct, os.path.join(save_dir, "rf_correct.joblib"))
    joblib.dump(models.rf_conf, os.path.join(save_dir, "rf_conf.joblib"))
    print(f"[Models] Saved -> {save_dir}")


def load_proxy_models(save_dir: str) -> ProxyModels:
    return ProxyModels(
        rf_iees=joblib.load(os.path.join(save_dir, "rf_iees.joblib")),
        rf_correct=joblib.load(os.path.join(save_dir, "rf_correct.joblib")),
        rf_conf=joblib.load(os.path.join(save_dir, "rf_conf.joblib")),
    )


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 3 and X.shape[-1] == 1:
        X = X.squeeze(-1)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}")
    if X.shape[1] != 10:
        raise ValueError(f"Expected 10 proxy features, got {X.shape[1]}")
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Runtime policy evaluation: attribution-free
# ---------------------------------------------------------------------------

def _runtime_exit_feature(model, x: torch.Tensor, exit_idx: int, prev_state: Optional[dict]):
    """Forward one exit and return 10-feature vector plus prediction."""
    logits, class_idx, _ = model.forward_to_exit(x, exit_idx, False)
    feature, state, conf, pred, _entropy = _feature_from_logits_and_activation(
        logits, getattr(model, "activations", None), prev_state
    )
    return feature.reshape(1, -1), state, conf, pred


def _score_proxy(model_name: str, models: ProxyModels, feat: np.ndarray) -> float:
    if model_name == "rf_iees":
        return float(models.rf_iees.predict(feat)[0])
    if model_name == "rf_correct":
        return float(models.rf_correct.predict_proba(feat)[0, 1])
    if model_name == "rf_conf":
        return float(models.rf_conf.predict(feat)[0])
    raise ValueError(f"Unknown proxy model name: {model_name}")


def evaluate_policy(
    model,
    dataloader,
    device: torch.device | str,
    policy: str,
    thresholds: Sequence[float],
    proxy_models: Optional[ProxyModels] = None,
) -> Dict[str, object]:
    """
    Evaluate one attribution-free runtime policy.

    policy ∈ {'confidence_only', 'rf_iees', 'rf_correct', 'rf_conf'}.
    """
    device = torch.device(device)
    model.eval()
    num_exits = _safe_num_exits(model)

    correct: List[int] = []
    exit_idx_list: List[int] = []
    latency_ms: List[float] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            for sample_idx in range(inputs.size(0)):
                x = inputs[sample_idx : sample_idx + 1]
                y = int(labels[sample_idx].item())
                prev_state: Optional[dict] = None
                final_pred = None
                chosen_exit = num_exits - 1

                t0 = time.perf_counter()
                for exit_idx in range(num_exits):
                    feat, prev_state, conf, pred = _runtime_exit_feature(
                        model, x, exit_idx, prev_state
                    )
                    final_pred = pred

                    if policy == "confidence_only":
                        score = conf
                    else:
                        if proxy_models is None:
                            raise ValueError("proxy_models required for RF policies")
                        score = _score_proxy(policy, proxy_models, feat)

                    if exit_idx < len(thresholds) and score >= thresholds[exit_idx]:
                        chosen_exit = exit_idx
                        break

                    if exit_idx == num_exits - 1:
                        chosen_exit = exit_idx
                        break

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                correct.append(int(final_pred == y))
                exit_idx_list.append(chosen_exit)
                latency_ms.append((t1 - t0) * 1000.0)

    exit_rates, exit_accs = compute_per_exit_stats(correct, exit_idx_list, num_exits)
    return {
        "correct": correct,
        "exit_idx": exit_idx_list,
        "time": latency_ms,
        "exit_rate": exit_rates,
        "exit_acc": exit_accs,
        "overall_acc": 100.0 * float(np.mean(correct)),
        "avg_time_ms": float(np.mean(latency_ms)),
    }


def tune_thresholds(
    model,
    proxy_models: Optional[ProxyModels],
    val_loader,
    device: torch.device | str,
    policy: str,
    threshold_vals: Sequence[float],
    calibration_mode: str = "shared",
) -> List[float]:
    """Tune thresholds on validation data only."""
    num_exits = _safe_num_exits(model)
    best_acc = -1.0
    best_thresholds: Optional[List[float]] = None

    if calibration_mode == "shared":
        candidates = [[float(t)] * num_exits for t in threshold_vals]
    elif calibration_mode == "per_exit":
        candidates = [list(map(float, ts)) for ts in product(threshold_vals, repeat=num_exits)]
    else:
        raise ValueError("calibration_mode must be 'shared' or 'per_exit'")

    print(f"Tuning {policy}: {len(candidates)} threshold candidates on validation")
    for thresholds in candidates:
        result = evaluate_policy(
            model=model,
            dataloader=val_loader,
            device=device,
            policy=policy,
            thresholds=thresholds,
            proxy_models=proxy_models,
        )
        acc = float(result["overall_acc"])
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds
            print(f"  new best {policy}: {best_acc:.2f}% @ {best_thresholds}")

    assert best_thresholds is not None
    return best_thresholds


def tune_matched_confidence_thresholds(
    model,
    val_loader,
    device: torch.device | str,
    target_exit_dist: Sequence[float],
    threshold_vals: Sequence[float],
    calibration_mode: str = "shared",
) -> List[float]:
    """Tune confidence thresholds on validation to match a target exit distribution."""
    num_exits = _safe_num_exits(model)

    if calibration_mode == "shared":
        candidates = [[float(t)] * num_exits for t in threshold_vals]
    elif calibration_mode == "per_exit":
        candidates = [list(map(float, ts)) for ts in product(threshold_vals, repeat=num_exits)]
    else:
        raise ValueError("calibration_mode must be 'shared' or 'per_exit'")

    best_thresholds: Optional[List[float]] = None
    best_diff = float("inf")

    print("Tuning Conf. (IEES-matched) on validation only")
    for thresholds in candidates:
        result = evaluate_policy(
            model=model,
            dataloader=val_loader,
            device=device,
            policy="confidence_only",
            thresholds=thresholds,
            proxy_models=None,
        )
        dist = result["exit_rate"]
        diff = sum(abs(float(dist[i]) - float(target_exit_dist[i])) for i in range(num_exits))
        if diff < best_diff:
            best_diff = diff
            best_thresholds = thresholds

    assert best_thresholds is not None
    print(f"  best matched confidence thresholds: {best_thresholds} | diff={best_diff:.2f}")
    return best_thresholds


# ---------------------------------------------------------------------------
# Table and case analysis
# ---------------------------------------------------------------------------

def compute_per_exit_stats(
    correct_list: Sequence[int],
    exit_idx_list: Sequence[int],
    num_exits: int,
) -> Tuple[List[float], List[Optional[float]]]:
    total = len(correct_list)
    exit_rates: List[float] = []
    exit_accs: List[Optional[float]] = []

    for e in range(num_exits):
        idxs = [i for i, exit_idx in enumerate(exit_idx_list) if exit_idx == e]
        exit_rates.append(100.0 * len(idxs) / total if total else 0.0)
        if idxs:
            exit_accs.append(100.0 * sum(correct_list[i] for i in idxs) / len(idxs))
        else:
            exit_accs.append(None)
    return exit_rates, exit_accs


def print_proxy_comparison_table(results: Dict[str, Dict[str, object]], num_exits: int) -> None:
    method_names = {
        "confidence_only": "Confidence only",
        "rf_conf": "Conf. Recalib RF",
        "rf_correct": "Correctness RF",
        "rf_iees": "IEES RF (Ours)",
        "conf_iees_matched": "Conf. (IEES-matched)",
    }
    order = [
        "confidence_only",
        "rf_conf",
        "rf_correct",
        "rf_iees",
        "conf_iees_matched",
    ]
    order = [k for k in order if k in results]

    print("\n" + "=" * 120)
    print("Proxy Supervision Comparison — 10-feature attribution-free runtime policies")
    print("=" * 120)

    print(f"{'Policy':<24}", end="")
    for i in range(num_exits):
        print(f" E{i + 1}% ", end="")
    print("  ", end="")
    for i in range(num_exits):
        print(f"E{i + 1}Acc ", end="")
    print(f" {'Overall':>10} {'Time':>10}")
    print("-" * 120)

    for key in order:
        r = results[key]
        print(f"{method_names[key]:<24}", end="")
        for v in r["exit_rate"]:
            print(f" {float(v):5.1f}", end="")
        print("  ", end="")
        for v in r["exit_acc"]:
            if v is None:
                print("     – ", end="")
            else:
                print(f" {float(v):6.1f}", end="")
        print(f" {float(r['overall_acc']):9.2f} {float(r['avg_time_ms']):8.2f}ms")

    print("=" * 120)


def case_analysis_lucky_correct(
    eval_pack: ProxyFeaturePack,
    threshold: float = 0.4,
) -> Dict[str, float]:
    """
    Count samples/exits where correctness=1 but oracle IEES is low.
    These are 'lucky correct' cases that motivate semantic reliability.
    """
    mask = (eval_pack.y_correct.astype(int) == 1) & (eval_pack.y_iees < threshold)
    count = int(mask.sum())
    pct = 100.0 * count / len(eval_pack.y_correct)
    print("\nCase analysis: correctness=1 but low IEES")
    print(f"  threshold={threshold:.2f}, count={count}, pct={pct:.2f}%")
    return {"threshold": threshold, "count": count, "pct": pct}


# ---------------------------------------------------------------------------
# End-to-end ablation driver
# ---------------------------------------------------------------------------

def run_proxy_supervision_comparison(
    model,
    train_or_cal_loader,
    val_loader,
    test_loader,
    device: torch.device | str,
    weights: Sequence[float],
    train_cache_path: Optional[str] = "cache/proxy_train_10feature.npz",
    val_cache_path: Optional[str] = "cache/proxy_val_10feature.npz",
    threshold_grid: Optional[Sequence[float]] = None,
    calibration_mode: str = "shared",
    save_json_path: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Full paper-synced proxy comparison.

    Steps
    -----
    1. Extract/load oracle IEES targets and 10-feature proxy inputs.
    2. Train IEES, correctness, and confidence RF baselines.
    3. Tune all thresholds on validation only.
    4. Evaluate on test using attribution-free runtime features only.
    5. Print a rebuttal/appendix-ready table.
    """
    if threshold_grid is None:
        threshold_grid = np.linspace(0.50, 0.95, 10)

    # 1. Extract/load training/calibration features.
    if train_cache_path and os.path.exists(train_cache_path):
        train_pack = load_proxy_features(train_cache_path)
    else:
        train_pack = extract_proxy_features_10_oracle_iees(
            model,
            train_or_cal_loader,
            device,
            weights,
            desc="Extracting train/calibration proxy pack",
        )
        if train_cache_path:
            save_proxy_features(train_cache_path, train_pack)

    if val_cache_path and os.path.exists(val_cache_path):
        val_pack = load_proxy_features(val_cache_path)
    else:
        val_pack = extract_proxy_features_10_oracle_iees(
            model,
            val_loader,
            device,
            weights,
            desc="Extracting validation proxy pack",
        )
        if val_cache_path:
            save_proxy_features(val_cache_path, val_pack)

    # 2. Train RFs and report fidelity on validation pack.
    models = train_proxy_supervision_baselines(train_pack)
    report_proxy_fidelity(models, val_pack)
    case_analysis_lucky_correct(val_pack, threshold=0.4)

    # 3. Tune thresholds on validation only.
    thresholds_conf = tune_thresholds(
        model,
        None,
        val_loader,
        device,
        policy="confidence_only",
        threshold_vals=threshold_grid,
        calibration_mode=calibration_mode,
    )
    thresholds_iees = tune_thresholds(
        model,
        models,
        val_loader,
        device,
        policy="rf_iees",
        threshold_vals=threshold_grid,
        calibration_mode=calibration_mode,
    )
    thresholds_correct = tune_thresholds(
        model,
        models,
        val_loader,
        device,
        policy="rf_correct",
        threshold_vals=threshold_grid,
        calibration_mode=calibration_mode,
    )
    thresholds_rf_conf = tune_thresholds(
        model,
        models,
        val_loader,
        device,
        policy="rf_conf",
        threshold_vals=threshold_grid,
        calibration_mode=calibration_mode,
    )

    # Validation IEES distribution for matched-confidence baseline.
    val_iees_result = evaluate_policy(
        model,
        val_loader,
        device,
        policy="rf_iees",
        thresholds=thresholds_iees,
        proxy_models=models,
    )
    thresholds_matched = tune_matched_confidence_thresholds(
        model,
        val_loader,
        device,
        target_exit_dist=val_iees_result["exit_rate"],
        threshold_vals=threshold_grid,
        calibration_mode=calibration_mode,
    )

    # 4. Evaluate on test, attribution-free.
    results: Dict[str, Dict[str, object]] = {}
    results["confidence_only"] = evaluate_policy(
        model, test_loader, device, "confidence_only", thresholds_conf, None
    )
    results["rf_conf"] = evaluate_policy(
        model, test_loader, device, "rf_conf", thresholds_rf_conf, models
    )
    results["rf_correct"] = evaluate_policy(
        model, test_loader, device, "rf_correct", thresholds_correct, models
    )
    results["rf_iees"] = evaluate_policy(
        model, test_loader, device, "rf_iees", thresholds_iees, models
    )
    results["conf_iees_matched"] = evaluate_policy(
        model, test_loader, device, "confidence_only", thresholds_matched, None
    )

    num_exits = _safe_num_exits(model)
    print_proxy_comparison_table(results, num_exits=num_exits)

    # 5. Save compact JSON if requested.
    if save_json_path:
        dirname = os.path.dirname(save_json_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        serializable = {
            "thresholds": {
                "confidence_only": thresholds_conf,
                "rf_conf": thresholds_rf_conf,
                "rf_correct": thresholds_correct,
                "rf_iees": thresholds_iees,
                "conf_iees_matched": thresholds_matched,
            },
            "results": {
                k: {
                    "exit_rate": v["exit_rate"],
                    "exit_acc": v["exit_acc"],
                    "overall_acc": v["overall_acc"],
                    "avg_time_ms": v["avg_time_ms"],
                }
                for k, v in results.items()
            },
        }
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"[Results] Saved -> {save_json_path}")

    return results


__all__ = [
    "ProxyFeaturePack",
    "ProxyModels",
    "save_proxy_features",
    "load_proxy_features",
    "extract_proxy_features_10_oracle_iees",
    "train_proxy_supervision_baselines",
    "report_proxy_fidelity",
    "evaluate_policy",
    "tune_thresholds",
    "tune_matched_confidence_thresholds",
    "print_proxy_comparison_table",
    "case_analysis_lucky_correct",
    "run_proxy_supervision_comparison",
]
