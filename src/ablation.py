from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional
from src.train import evaluate_model_thresholds
import pandas as pd


def _build_variants_from_weights(base_w: List[float]) -> Dict[str, List[float]]:
    if len(base_w) != 3:
        return {"IEES_full": copy.deepcopy(base_w)}
    wS, wF, wQ = base_w
    return {
        "IEES_full (S+F-Q)": [wS, wF, wQ],
        "w/o confidence S": [0.0, wF, wQ],
        "w/o attribution strength F": [wS, 0.0, wQ],
        "w/o stability penalty Q": [wS, wF, 0.0],
        "confidence only (S)": [wS, 0.0, 0.0],
        "attribution strength only (F)": [0.0, wF, 0.0],
        "confidence + strength (S+F)": [wS, wF, 0.0],
    }


def run_iees_ablation(model, dataloader, device, out_csv: Optional[str] = None,
                      taus: Optional[List[float]] = None, approx_flops_per_exit=None,
                      max_samples: Optional[int] = None, base_weights: Optional[List[float]] = None,
                      model_name: Optional[str] = None):
    rows: List[Dict[str, Any]] = []
    conf_res = evaluate_model_thresholds(model, model_name, dataloader, device, "confidence", taus, weights=[])
    rows += [{"variant": "Confidence-only", **r} for r in conf_res]
    if base_weights is not None:
        for vname, vweights in _build_variants_from_weights(list(base_weights)).items():
            res = evaluate_model_thresholds(model, model_name, dataloader, device, "iees", taus, weights=vweights, proxy=None)
            rows += [{"variant": vname, **r} for r in res]
    if pd is not None:
        df = pd.DataFrame(rows)
        if out_csv:
            df.to_csv(out_csv, index=False)
        return df
    return rows
