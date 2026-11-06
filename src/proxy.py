from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

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
    generate_pfams,
    minmax_normalize,
)


Array = np.ndarray
Weights = Union[Tuple[float, float, float], List[float]]


def get_num_decision_exits(model) -> int:
    """Return the number of decision exits including the final classifier.

    Some repository models store only side exits in ``model.num_exits`` and
    expose the final classifier as ``exit_idx == len(model.exits)``. Other
    external frameworks (e.g., DVT/MSDNet wrappers) already expose all decision
    points through ``num_exits``. This helper keeps proxy extraction consistent
    with the paper tables, where E1..Ek include the final exit when applicable.
    """
    exits = getattr(model, 'exits', None)
    out_put = getattr(model, 'out_put', None)
    final_classifier = getattr(model, 'final_classifier', None)
    if exits is not None and (out_put is not None or final_classifier is not None):
        return len(exits) + 1
    return int(getattr(model, 'num_exits'))


@dataclass
class ProxyFeatureState:
    """Previous-exit state needed for cross-exit proxy features."""

    conf: float
    pred: int
    logits: torch.Tensor
    entropy: float
    act_mean: float


@dataclass
class ProxyExtractionOutput:
    """Container for extracted proxy features and supervision targets."""

    X: Array
    y_iees: Array
    y_correct: Array
    y_conf: Array
    y_F_raw: Array
    y_Q_raw: Array
    exits: Array


def _safe_numpy_scalar(x: Union[float, torch.Tensor, Array]) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    if isinstance(x, np.ndarray):
        return float(np.asarray(x).reshape(-1)[0])
    return float(x)


def compute_attribution_strength_from_map(channel_weighted_map: Array, k: int = 20) -> float:
    """
    Paper Eq. 7--8: F_i(x) from the raw, pre-ReLU channel-weighted map M_i(x).

    F_i is the mean value over the top-k% most discriminative spatial locations.
    The map must NOT be ReLU-normalized before this calculation.
    """
    m = np.asarray(channel_weighted_map, dtype=np.float64)
    if m.size == 0:
        return 0.0
    q = np.quantile(m, 1.0 - k / 100.0)
    mask = m >= q
    if not np.any(mask):
        return 0.0
    return float(m[mask].mean())


def compute_forward_features_from_logits(
    logits: torch.Tensor,
    activations: Optional[torch.Tensor],
    prev_state: Optional[ProxyFeatureState],
) -> Tuple[Array, ProxyFeatureState]:
    """
    Build the 10-feature attribution-free proxy vector for one exit.

    Feature order:
      1 confidence
      2 margin
      3 entropy
      4 activation_mean
      5 activation_max
      6 confidence_delta
      7 class_stability
      8 logit_cosine
      9 entropy_delta
      10 activation_delta
    """
    probs = F.softmax(logits, dim=1)

    conf = float(probs.max(dim=1).values.item())
    entropy = float(-(probs * torch.log(probs + 1e-8)).sum(dim=1).item())
    top2 = torch.topk(probs, 2, dim=1).values
    margin = float((top2[:, 0] - top2[:, 1]).item())
    pred = int(probs.argmax(dim=1).item())

    if activations is not None:
        act_mean = float(activations.detach().mean().item())
        act_max = float(activations.detach().amax().item())
    else:
        # Fallback for models where activation capture is unavailable.
        act_mean = conf
        act_max = conf

    if prev_state is None:
        conf_delta = 0.0
        class_stability = 0.0
        logit_cos = 0.0
        entropy_delta = 0.0
        act_delta = 0.0
    else:
        conf_delta = abs(conf - prev_state.conf)
        class_stability = float(pred == prev_state.pred)
        logit_cos = float(
            F.cosine_similarity(
                logits.detach().flatten(start_dim=1),
                prev_state.logits.detach().flatten(start_dim=1),
                dim=1,
            ).item()
        )
        entropy_delta = abs(entropy - prev_state.entropy)
        act_delta = abs(act_mean - prev_state.act_mean)

    features = np.array(
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

    state = ProxyFeatureState(
        conf=conf,
        pred=pred,
        logits=logits.detach(),
        entropy=entropy,
        act_mean=act_mean,
    )
    return features, state


def _oracle_components_for_sample(
    model,
    input_image: torch.Tensor,
    label: torch.Tensor,
    k: int,
    common_resolution: Tuple[int, int],
) -> Tuple[List[Array], List[float], List[float], List[float], List[int], List[float]]:
    """
    Compute raw oracle components per exit for one sample.

    Returns lists for features, S, F_raw, Q_raw, correctness, confidence.
    IEES is intentionally not computed here because F and Q must be normalized
    across the calibration/validation extraction split before applying Eq. 5.
    """
    features_per_exit: List[Array] = []
    S_values: List[float] = []
    F_raw_values: List[float] = []
    Q_raw_values: List[float] = []
    correctness_values: List[int] = []
    conf_values: List[float] = []

    pfams_list: List[Array] = []
    cumulative_maps: List[Array] = []
    channel_weighted_maps: List[Array] = []
    prev_state: Optional[ProxyFeatureState] = None

    num_exits = get_num_decision_exits(model)

    for exit_idx in range(num_exits):
        model.zero_grad(set_to_none=True)

        logits, _class_idx, _target_score = model.forward_to_exit(
            input_image, exit_idx, True
        )

        probs = F.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        correct = int(pred == int(label.item()))
        S = float(probs.max(dim=1).values.item())

        pfams_list, cumulative_map, channel_weighted_maps = generate_pfams(
            model.activations,
            model.gradients,
            input_image,
            pfams_list,
            channel_weighted_maps,
            common_resolution=common_resolution,
        )
        cumulative_maps.append(cumulative_map)

        Q_raw = float(calculate_consistency_index(cumulative_maps))
        F_raw = compute_attribution_strength_from_map(
            channel_weighted_maps[-1], k=k
        )

        feat, prev_state = compute_forward_features_from_logits(
            logits, model.activations, prev_state
        )

        features_per_exit.append(feat)
        S_values.append(S)
        F_raw_values.append(F_raw)
        Q_raw_values.append(Q_raw)
        correctness_values.append(correct)
        conf_values.append(S)

        model.activations = None
        model.gradients = None

    return (
        features_per_exit,
        S_values,
        F_raw_values,
        Q_raw_values,
        correctness_values,
        conf_values,
    )


def compute_iees_from_components(
    S: Array,
    F_raw: Array,
    Q_raw: Array,
    weights: Weights,
    normalize_components: bool = True,
) -> Tuple[Array, Array, Array]:
    """
    Apply paper Eq. 5 using normalized F and Q by default.

      IEES_i(x) = w1*S_i(x) + w2*F_i(x) - w3*Q_i(x)

    S is already in [0,1]. F and Q are min-max normalized across the extracted
    split unless normalize_components=False is explicitly requested.
    """
    S_arr = np.asarray(S, dtype=np.float64)
    F_arr = np.asarray(F_raw, dtype=np.float64)
    Q_arr = np.asarray(Q_raw, dtype=np.float64)

    if normalize_components:
        F_used = minmax_normalize(F_arr)
        Q_used = minmax_normalize(Q_arr)
    else:
        F_used = F_arr
        Q_used = Q_arr

    w1, w2, w3 = [float(v) for v in weights]
    y_iees = w1 * S_arr + w2 * F_used - w3 * Q_used
    return y_iees.astype(np.float32), F_used.astype(np.float32), Q_used.astype(np.float32)


def extract_proxy_features(
    model,
    dataloader,
    device: Union[str, torch.device],
    weights: Weights,
    k: int = 20,
    common_resolution: Tuple[int, int] = DEFAULT_COMMON_RESOLUTION,
    normalize_components: bool = True,
    desc: str = "Extracting proxy features",
) -> ProxyExtractionOutput:
    """
    Extract 10-feature proxy inputs and oracle IEES targets.

    This is the paper-aligned offline step. It computes PFAM/Grad-CAM only to
    create oracle supervision targets; the returned X contains no attribution maps.
    """
    device = torch.device(device)
    model.eval()

    X_list: List[Array] = []
    S_list: List[float] = []
    F_raw_list: List[float] = []
    Q_raw_list: List[float] = []
    y_correct_list: List[int] = []
    y_conf_list: List[float] = []
    exit_list: List[int] = []

    print(f"{desc}...")

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}")

        inputs = inputs.to(device)
        labels = labels.to(device)

        for sample_idx in range(inputs.size(0)):
            input_image = inputs[sample_idx : sample_idx + 1]
            label = labels[sample_idx]

            (
                feats,
                S_vals,
                F_vals,
                Q_vals,
                corr_vals,
                conf_vals,
            ) = _oracle_components_for_sample(
                model,
                input_image,
                label,
                k=k,
                common_resolution=common_resolution,
            )

            for exit_idx, feat in enumerate(feats):
                X_list.append(feat)
                S_list.append(S_vals[exit_idx])
                F_raw_list.append(F_vals[exit_idx])
                Q_raw_list.append(Q_vals[exit_idx])
                y_correct_list.append(corr_vals[exit_idx])
                y_conf_list.append(conf_vals[exit_idx])
                exit_list.append(exit_idx)

    X = np.asarray(X_list, dtype=np.float32)
    S = np.asarray(S_list, dtype=np.float32)
    F_raw = np.asarray(F_raw_list, dtype=np.float32)
    Q_raw = np.asarray(Q_raw_list, dtype=np.float32)
    y_correct = np.asarray(y_correct_list, dtype=np.int32)
    y_conf = np.asarray(y_conf_list, dtype=np.float32)
    exits = np.asarray(exit_list, dtype=np.int32)

    y_iees, F_used, Q_used = compute_iees_from_components(
        S, F_raw, Q_raw, weights, normalize_components=normalize_components
    )

    print(f"  Extracted {len(X)} exit-samples with {X.shape[1]} features")
    print(f"  IEES range: [{y_iees.min():.3f}, {y_iees.max():.3f}]")

    return ProxyExtractionOutput(
        X=X,
        y_iees=y_iees,
        y_correct=y_correct,
        y_conf=y_conf,
        y_F_raw=F_raw,
        y_Q_raw=Q_raw,
        exits=exits,
    )


def train_iees_proxy_rf(
    X_train: Array,
    y_train: Array,
    save_path: Optional[str] = None,
    n_estimators: int = 100,
    max_depth: int = 8,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train the default RF regressor used for proxy IEES."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(rf, save_path)
    return rf


def train_confidence_recalib_rf(
    X_train: Array,
    y_conf_train: Array,
    save_path: Optional[str] = None,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Baseline: RF trained to predict confidence only."""
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_conf_train)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(rf, save_path)
    return rf


def train_correctness_rf(
    X_train: Array,
    y_correct_train: Array,
    save_path: Optional[str] = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Baseline: RF classifier trained to predict binary correctness."""
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_correct_train.astype(int))
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(rf, save_path)
    return rf


def evaluate_regression_proxy(model, X: Array, y: Array, name: str = "IEES RF") -> Dict[str, float]:
    """Return standard proxy fidelity metrics."""
    pred = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    metrics = {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": rmse,
    }
    print(f"\n=== {name} ===")
    print(f"R²:   {metrics['r2']:.3f}")
    print(f"MAE:  {metrics['mae']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    return metrics


def evaluate_correctness_proxy(model, X: Array, y_correct: Array, name: str = "Correctness RF") -> Dict[str, float]:
    """Return correctness-classifier metrics."""
    pred = model.predict(X)
    metrics = {"accuracy": float(accuracy_score(y_correct, pred))}
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_correct, prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    print(f"\n=== {name} ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC:  {metrics['roc_auc']:.3f}")
    return metrics


def train_all_proxy_baselines(
    train_data: ProxyExtractionOutput,
    test_data: ProxyExtractionOutput,
    save_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    Train IEES RF and the paper's supervision baselines using identical features.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    rf_iees = train_iees_proxy_rf(
        train_data.X,
        train_data.y_iees,
        save_path=os.path.join(save_dir, "rf_iees.joblib") if save_dir else None,
    )
    rf_conf = train_confidence_recalib_rf(
        train_data.X,
        train_data.y_conf,
        save_path=os.path.join(save_dir, "rf_confidence.joblib") if save_dir else None,
    )
    rf_corr = train_correctness_rf(
        train_data.X,
        train_data.y_correct,
        save_path=os.path.join(save_dir, "rf_correctness.joblib") if save_dir else None,
    )

    metrics = {
        "iees": evaluate_regression_proxy(rf_iees, test_data.X, test_data.y_iees, "IEES RF"),
        "confidence": evaluate_regression_proxy(
            rf_conf, test_data.X, test_data.y_conf, "Confidence Recalib RF"
        ),
        "correctness": evaluate_correctness_proxy(
            rf_corr, test_data.X, test_data.y_correct, "Correctness RF"
        ),
    }

    return {
        "rf_iees": rf_iees,
        "rf_confidence": rf_conf,
        "rf_correctness": rf_corr,
        "metrics": metrics,
    }


def proxy_features_for_runtime_exit(
    logits: torch.Tensor,
    activations: Optional[torch.Tensor],
    prev_state: Optional[ProxyFeatureState],
) -> Tuple[Array, ProxyFeatureState]:
    """
    Runtime feature extraction: no gradients, no PFAM, no attribution maps.
    """
    return compute_forward_features_from_logits(logits, activations, prev_state)


def inference_with_iees_proxy(
    model,
    rf_proxy: RandomForestRegressor,
    thresholds: Iterable[float],
    x: torch.Tensor,
    device: Union[str, torch.device],
):
    """
    Runtime proxy IEES inference.

    This function intentionally does NOT compute Grad-CAM/PFAM. It only uses
    the 10 attribution-free features and the trained RF proxy.
    """
    device = torch.device(device)
    x = x.to(device)
    thresholds = list(thresholds)
    prev_state: Optional[ProxyFeatureState] = None

    model.eval()
    with torch.no_grad():
        num_exits = get_num_decision_exits(model)
        for exit_idx in range(num_exits):
            logits, _class_idx, _target_score = model.forward_to_exit(
                x, exit_idx, False
            )
            activations = getattr(model, "activations", None)
            feat, prev_state = proxy_features_for_runtime_exit(
                logits, activations, prev_state
            )
            pred_iees = float(rf_proxy.predict(feat.reshape(1, -1))[0])

            if pred_iees >= thresholds[exit_idx] or exit_idx == num_exits - 1:
                return logits, exit_idx, pred_iees

    # Unreachable, but kept for type safety.
    return logits, num_exits - 1, pred_iees


def save_extraction_output(path: str, data: ProxyExtractionOutput) -> None:
    """Persist extracted proxy arrays."""
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "X.npy"), data.X)
    np.save(os.path.join(path, "y_iees.npy"), data.y_iees)
    np.save(os.path.join(path, "y_correct.npy"), data.y_correct)
    np.save(os.path.join(path, "y_conf.npy"), data.y_conf)
    np.save(os.path.join(path, "y_F_raw.npy"), data.y_F_raw)
    np.save(os.path.join(path, "y_Q_raw.npy"), data.y_Q_raw)
    np.save(os.path.join(path, "exits.npy"), data.exits)


def load_extraction_output(path: str) -> ProxyExtractionOutput:
    """Load extracted proxy arrays."""
    return ProxyExtractionOutput(
        X=np.load(os.path.join(path, "X.npy")),
        y_iees=np.load(os.path.join(path, "y_iees.npy")),
        y_correct=np.load(os.path.join(path, "y_correct.npy")),
        y_conf=np.load(os.path.join(path, "y_conf.npy")),
        y_F_raw=np.load(os.path.join(path, "y_F_raw.npy")),
        y_Q_raw=np.load(os.path.join(path, "y_Q_raw.npy")),
        exits=np.load(os.path.join(path, "exits.npy")),
    )



def train_proxy(model, val_loader, test_loader=None, weights=(0.5, 0.3, 0.2), device=None, save_path=None):
    """Backward-compatible entry point used by main.py.

    Extracts paper-aligned 10-feature proxy inputs from ``val_loader`` and
    trains the default Random Forest IEES proxy. If ``test_loader`` is supplied,
    it reports proxy fidelity on that split. Runtime inference should use
    ``inference_with_iees_proxy`` and does not compute PFAM/Grad-CAM.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = extract_proxy_features(model, val_loader, device, weights, desc="Proxy training extraction")
    rf = train_iees_proxy_rf(train_data.X, train_data.y_iees, save_path=save_path)
    # if test_loader is not None:
    #     test_data = extract_proxy_features(model, test_loader, device, weights, desc="Proxy fidelity extraction")
    #     evaluate_regression_proxy(rf, test_data.X, test_data.y_iees, "IEES RF")
    return rf
