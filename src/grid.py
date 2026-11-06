from __future__ import annotations
from itertools import product
from typing import Iterable, Optional
import numpy as np
import torch
import torch.nn.functional as F
from src.proxy import ProxyFeatureState, get_num_decision_exits, proxy_features_for_runtime_exit


def _as_thresholds(thresholds, n):
    if isinstance(thresholds, (float, int)):
        return [float(thresholds)] * n
    thresholds = list(thresholds)
    if len(thresholds) < n:
        thresholds += [thresholds[-1]] * (n - len(thresholds))
    return thresholds[:n]


def evaluate_confidence(model, dataloader, device, mode="shared", *thresholds):
    model.eval(); n = get_num_decision_exits(model)
    thresholds = _as_thresholds(thresholds[0] if mode == "shared" else thresholds, n)
    correct = total = 0
    with torch.no_grad():
        for x, target in dataloader:
            x, target = x.to(device), target.to(device)
            for b in range(x.size(0)):
                img, y = x[b:b+1], int(target[b].item())
                pred = None
                for i in range(n):
                    logits, _, _ = model.forward_to_exit(img, i, False)
                    probs = F.softmax(logits, dim=1)
                    conf = float(probs.max(dim=1).values.item())
                    pred = int(probs.argmax(dim=1).item())
                    if conf >= thresholds[i] or i == n - 1:
                        break
                correct += int(pred == y); total += 1
    return correct / max(total, 1)


def evaluate_iees_proxy(model, rf_proxy, dataloader, device, mode="shared", *thresholds):
    model.eval(); n = get_num_decision_exits(model)
    thresholds = _as_thresholds(thresholds[0] if mode == "shared" else thresholds, n)
    correct = total = 0
    with torch.no_grad():
        for x, target in dataloader:
            x, target = x.to(device), target.to(device)
            for b in range(x.size(0)):
                img, y = x[b:b+1], int(target[b].item())
                prev_state: Optional[ProxyFeatureState] = None
                pred = None
                for i in range(n):
                    logits, _, _ = model.forward_to_exit(img, i, False)
                    feat, prev_state = proxy_features_for_runtime_exit(logits, getattr(model, 'activations', None), prev_state)
                    score = float(rf_proxy.predict(feat.reshape(1, -1))[0])
                    pred = int(F.softmax(logits, dim=1).argmax(dim=1).item())
                    if score >= thresholds[i] or i == n - 1:
                        break
                correct += int(pred == y); total += 1
    return correct / max(total, 1)


def grid_search_confidence(model, valloader, device, calibration_mode="shared", threshold_vals=None):
    threshold_vals = np.linspace(0.5, 0.95, 10) if threshold_vals is None else threshold_vals
    n = get_num_decision_exits(model)
    best_acc, best_config = -1.0, {}
    if calibration_mode == "shared":
        for tau in threshold_vals:
            acc = evaluate_confidence(model, valloader, device, "shared", float(tau))
            if acc > best_acc:
                best_acc, best_config = acc, {"thresholds": [float(tau)] * n}
    else:
        for taus in product(threshold_vals, repeat=n):
            acc = evaluate_confidence(model, valloader, device, "per_exit", *[float(t) for t in taus])
            if acc > best_acc:
                best_acc, best_config = acc, {"thresholds": [float(t) for t in taus]}
    print(f"[GRID] Best confidence val accuracy={best_acc:.4f}; config={best_config}")
    return best_config


def grid_search_iees(model, valloader, device, rf_proxy=None, calibration_mode="shared", threshold_vals=None):
    if rf_proxy is None:
        raise ValueError("grid_search_iees now calibrates proxy IEES thresholds; pass rf_proxy trained on oracle IEES targets.")
    threshold_vals = np.linspace(0.5, 0.95, 10) if threshold_vals is None else threshold_vals
    n = get_num_decision_exits(model)
    best_acc, best_config = -1.0, {}
    if calibration_mode == "shared":
        for tau in threshold_vals:
            acc = evaluate_iees_proxy(model, rf_proxy, valloader, device, "shared", float(tau))
            if acc > best_acc:
                best_acc, best_config = acc, {"thresholds": [float(tau)] * n}
    else:
        for taus in product(threshold_vals, repeat=n):
            acc = evaluate_iees_proxy(model, rf_proxy, valloader, device, "per_exit", *[float(t) for t in taus])
            if acc > best_acc:
                best_acc, best_config = acc, {"thresholds": [float(t) for t in taus]}
    print(f"[GRID] Best IEES-proxy val accuracy={best_acc:.4f}; config={best_config}")
    return best_config
