"""
agnostic_updated_final.py

Attribution-method sensitivity utilities for PFAM/IEES.

This file updates the older IG/Grad-CAM helper so that it is consistent with the
final paper formulation:

    PFAM: exit-wise maps A_i and cumulative maps C_i at a common resolution
    Q_i : mean absolute change between consecutive cumulative maps
    IEES: w1*S_i + w2*F_i - w3*Q_i

Default / paper path:
    Grad-CAM, with F_i computed from the raw pre-ReLU channel-weighted map M_i.

Ablation path:
    Integrated Gradients (IG), with PFAM generated from normalized IG maps and
    F_i computed from a signed IG spatial attribution map over the top-k% salient
    locations. Use this only for attribution-method sensitivity, not as the default
    Eq. (6) Grad-CAM implementation.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_COMMON_RESOLUTION: Tuple[int, int] = (224, 224)
AttributionMethod = Literal["gradcam", "ig"]


def _minmax_np(arr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return ((arr - arr.min()) / (arr.max() - arr.min() + eps)).astype(np.float32)


def _resize_2d_np(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a 2-D numpy map to `size` using bilinear interpolation."""
    tensor = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).cpu().numpy()


# -----------------------------------------------------------------------------
# Grad-CAM path: final paper default
# -----------------------------------------------------------------------------
def generate_gradcam(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    common_resolution: Tuple[int, int] = DEFAULT_COMMON_RESOLUTION,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a Grad-CAM attribution map A_i and raw channel-weighted map M_i.

    Parameters
    ----------
    activations : torch.Tensor, shape (1, K, H, W)
        Feature maps Z_{i,k}(x) for exit i.
    gradients : torch.Tensor, shape (1, K, H, W)
        Gradients of the target class score with respect to activations.
    common_resolution : tuple
        Resolution used for all PFAM maps before cumulative aggregation.

    Returns
    -------
    cam : np.ndarray, shape common_resolution
        ReLU(M_i), min-max normalized to [0, 1]. This is A_i(x).
    channel_weighted_map : np.ndarray, shape common_resolution
        Raw pre-ReLU channel-weighted map M_i(x), used for F_i.
    """
    if activations is None or gradients is None:
        raise ValueError("Grad-CAM requires non-null activations and gradients.")

    alpha = gradients.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)
    raw_map = (alpha * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

    raw_up = F.interpolate(raw_map, size=common_resolution, mode="bilinear", align_corners=False)
    channel_weighted_map = raw_up.detach().squeeze().cpu().numpy()

    cam = torch.relu(raw_up).detach().squeeze().cpu().numpy()
    cam = _minmax_np(cam)
    return cam, channel_weighted_map.astype(np.float32)


def generate_pfams_gradcam(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    pfams_list: List[np.ndarray],
    channel_weighted_maps_list: List[np.ndarray],
    common_resolution: Tuple[int, int] = DEFAULT_COMMON_RESOLUTION,
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
    """
    Append Grad-CAM A_i and return cumulative PFAM C_i.

    This is the default paper-aligned PFAM path.
    """
    cam, channel_weighted_map = generate_gradcam(
        activations, gradients, common_resolution=common_resolution
    )
    pfams_list.append(cam)
    channel_weighted_maps_list.append(channel_weighted_map)
    cumulative_map = np.mean(np.stack(pfams_list, axis=0), axis=0).astype(np.float32)
    return pfams_list, cumulative_map, channel_weighted_maps_list


# -----------------------------------------------------------------------------
# Integrated Gradients path: attribution-method sensitivity / Appendix ablation
# -----------------------------------------------------------------------------
def generate_integrated_gradients(
    model,
    input_tensor: torch.Tensor,
    exit_idx: int,
    num_steps: int = 50,
    common_resolution: Tuple[int, int] = DEFAULT_COMMON_RESOLUTION,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an Integrated Gradients attribution map for one exit.

    This is intended for attribution-method sensitivity experiments. It assumes
    the model exposes `forward_to_exit(input_tensor, exit_idx, gradient_required)`.

    Returns
    -------
    ig_map : np.ndarray, shape common_resolution
        Non-negative normalized IG saliency map used as A_i for PFAM.
    signed_spatial_map : np.ndarray, shape common_resolution
        Signed spatial attribution map used for an IG-based F_i variant.
    """
    model.eval()
    if input_tensor.size(0) != 1:
        raise ValueError("Integrated Gradients path expects batch_size=1.")

    baseline = torch.zeros_like(input_tensor)
    alphas = torch.linspace(0.0, 1.0, num_steps + 1, device=input_tensor.device)
    total_grad = torch.zeros_like(input_tensor)

    # Use the predicted class at the actual input as the target for all steps.
    with torch.no_grad():
        logits, class_idx, _ = model.forward_to_exit(
            input_tensor, exit_idx, gradient_required=False
        )
        if torch.is_tensor(class_idx):
            target_class = int(class_idx.reshape(-1)[0].item())
        else:
            target_class = int(class_idx)

    for alpha in alphas:
        interp = (baseline + alpha * (input_tensor - baseline)).detach().requires_grad_(True)
        logits, _, _ = model.forward_to_exit(interp, exit_idx, gradient_required=False)
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        if interp.grad is not None:
            interp.grad.zero_()
        score.backward()
        if interp.grad is not None:
            total_grad += interp.grad.detach()

    # Average gradients along the path and multiply by input displacement.
    ig = (input_tensor - baseline) * total_grad / float(num_steps + 1)

    # Signed spatial map: sum signed attributions over channels.
    signed_map = ig.sum(dim=1, keepdim=True)  # (1, 1, H, W)
    signed_up = F.interpolate(
        signed_map, size=common_resolution, mode="bilinear", align_corners=False
    )
    signed_spatial_map = signed_up.detach().squeeze().cpu().numpy().astype(np.float32)

    # PFAM map: non-negative saliency, normalized.
    saliency = ig.abs().sum(dim=1, keepdim=True)
    saliency_up = F.interpolate(
        saliency, size=common_resolution, mode="bilinear", align_corners=False
    )
    ig_map = saliency_up.detach().squeeze().cpu().numpy()
    ig_map = _minmax_np(ig_map)
    return ig_map, signed_spatial_map


def generate_pfams_ig(
    model,
    input_tensor: torch.Tensor,
    exit_idx: int,
    pfams_list: List[np.ndarray],
    signed_maps_list: List[np.ndarray],
    num_steps: int = 50,
    common_resolution: Tuple[int, int] = DEFAULT_COMMON_RESOLUTION,
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
    """
    Append IG-based PFAM A_i and return cumulative PFAM C_i.

    Use this for attribution-method sensitivity only. The default paper path is
    `generate_pfams_gradcam`.
    """
    ig_map, signed_spatial_map = generate_integrated_gradients(
        model,
        input_tensor,
        exit_idx,
        num_steps=num_steps,
        common_resolution=common_resolution,
    )
    pfams_list.append(ig_map)
    signed_maps_list.append(signed_spatial_map)
    cumulative_map = np.mean(np.stack(pfams_list, axis=0), axis=0).astype(np.float32)
    return pfams_list, cumulative_map, signed_maps_list


# -----------------------------------------------------------------------------
# Shared final PFAM / IEES scoring
# -----------------------------------------------------------------------------
def calculate_consistency_index(cumulative_maps: Sequence[np.ndarray]) -> float:
    """
    Final paper Q_i: mean absolute change between C_i and C_{i-1}.

    Convention: Q_1 = 0. Smaller Q means more stable attribution focus.
    """
    if len(cumulative_maps) < 2:
        return 0.0

    curr = np.asarray(cumulative_maps[-1], dtype=np.float32)
    prev = np.asarray(cumulative_maps[-2], dtype=np.float32)
    if curr.shape != prev.shape:
        prev = _resize_2d_np(prev, curr.shape)
    return float(np.mean(np.abs(curr - prev)))


def compute_attribution_strength(spatial_map: np.ndarray, k: int = 20) -> float:
    """
    Compute F_i as the average value over the top-k% spatial locations.

    For Grad-CAM, `spatial_map` should be the raw pre-ReLU channel-weighted map
    M_i from Eq. (6). For IG sensitivity, pass the signed IG spatial attribution
    map and report the result as an attribution-method ablation rather than the
    default Eq. (6) Grad-CAM score.
    """
    spatial_map = np.asarray(spatial_map, dtype=np.float32)
    q = np.quantile(spatial_map, 1.0 - k / 100.0)
    mask = spatial_map >= q
    if mask.sum() == 0:
        return 0.0
    return float(spatial_map[mask].mean())


def minmax_normalize(values: Sequence[float], eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize scalar values to [0, 1]."""
    arr = np.asarray(values, dtype=np.float64)
    return ((arr - arr.min()) / (arr.max() - arr.min() + eps)).astype(np.float32)


def compute_iees_score_from_components(
    confidence: float,
    attribution_strength: float,
    instability: float,
    weights: Sequence[float],
) -> float:
    """Final IEES: w1*S + w2*F - w3*Q."""
    w1, w2, w3 = float(weights[0]), float(weights[1]), float(weights[2])
    return float(w1 * confidence + w2 * attribution_strength - w3 * instability)


def compute_iees_score(
    logits_or_probs: torch.Tensor,
    attribution_strength: float,
    instability: float,
    weights: Sequence[float],
    input_is_probs: bool = False,
) -> Tuple[float, float]:
    """
    Convenience wrapper for final IEES from model output and scalar F/Q.

    Parameters
    ----------
    logits_or_probs : torch.Tensor, shape (1, C)
        Raw logits by default. Set `input_is_probs=True` if passing softmax probabilities.
    attribution_strength : float
        F_i, preferably min-max normalized before final scoring.
    instability : float
        Q_i, preferably min-max normalized before final scoring.
    weights : sequence of float
        Non-negative weights [w1, w2, w3].
    input_is_probs : bool
        Whether `logits_or_probs` is already softmax probabilities.
    """
    probs = logits_or_probs if input_is_probs else F.softmax(logits_or_probs, dim=1)
    confidence = float(probs.max(dim=1).values.detach().cpu().item())
    iees = compute_iees_score_from_components(
        confidence, attribution_strength, instability, weights
    )
    return iees, confidence


# Backward-compatible alias. Prefer generate_pfams_gradcam in new code.
def generate_pfams(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    input_tensor: Optional[torch.Tensor],
    pfams_list: List[np.ndarray],
    channel_weighted_maps_list: Optional[List[np.ndarray]] = None,
    common_resolution: Tuple[int, int] = DEFAULT_COMMON_RESOLUTION,
):
    """
    Backward-compatible Grad-CAM PFAM wrapper.

    If `channel_weighted_maps_list` is omitted, returns the old 2-tuple
    `(pfams_list, cumulative_map)`. If provided, returns the final 3-tuple
    `(pfams_list, cumulative_map, channel_weighted_maps_list)`.
    """
    own_list = channel_weighted_maps_list is None
    if channel_weighted_maps_list is None:
        channel_weighted_maps_list = []
    out = generate_pfams_gradcam(
        activations,
        gradients,
        pfams_list,
        channel_weighted_maps_list,
        common_resolution=common_resolution,
    )
    if own_list:
        return out[0], out[1]
    return out


USAGE_NOTE = """
Recommended usage in final experiments:

1. Default paper-aligned Grad-CAM PFAM:

    pfams, cumulative, raw_maps = generate_pfams_gradcam(
        model.activations, model.gradients, pfams, raw_maps
    )
    cumulative_maps.append(cumulative)
    Q = calculate_consistency_index(cumulative_maps)
    F_i = compute_attribution_strength(raw_maps[-1], k=20)
    iees, S = compute_iees_score(logits, F_i, Q, weights)

2. IG sensitivity ablation:

    pfams, cumulative, ig_signed_maps = generate_pfams_ig(
        model, img, exit_idx, pfams, ig_signed_maps, num_steps=50
    )
    cumulative_maps.append(cumulative)
    Q = calculate_consistency_index(cumulative_maps)
    F_i = compute_attribution_strength(ig_signed_maps[-1], k=20)
    iees, S = compute_iees_score(logits, F_i, Q, weights)

For final reported deployment, use these attribution scores only offline to train
or audit the proxy. Runtime exiting should use attribution-free proxy features.
"""
