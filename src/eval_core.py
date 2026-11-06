import numpy as np, torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Iterable, List, Dict
from src.util import reorganize_attribution_maps
from src.xai import getAttribution, getAttribution_qty

# Optional deps (auto-fallbacks are provided)
_USE_SKIMAGE = True
try:
    from skimage.segmentation import slic
except Exception:
    _USE_SKIMAGE = False

try:
    import cv2
except Exception:
    cv2 = None


# ==============================
# Core helpers
# ==============================
def _to_hw(x) -> np.ndarray:
    """Accept [H,W], [1,H,W], [H,W,1], [C,H,W], [1,C,H,W] (np/torch); return float32 [H,W]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3:
        if x.shape[0] in (1, 3):             # CHW
            x = x[0] if x.shape[0] == 1 else x.mean(0)
        elif x.shape[-1] in (1, 3):           # HWC
            x = x[..., 0] if x.shape[-1] == 1 else x.mean(-1)
        else:
            x = x.mean(0)
    assert x.ndim == 2, f"Expected HxW, got {x.shape}"
    return x

def _minmax01(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mmin, mmax = float(m.min()), float(m.max())
    d = max(mmax - mmin, eps)
    return (m - mmin) / d

# --------- Normalization pipelines ---------
def _prep_maps_permap(exit_maps: List) -> List[np.ndarray]:
    """Per-exit min–max (used for faithfulness ranking/masking)."""
    return [_minmax01(_to_hw(m)) for m in exit_maps]

def _prep_maps_joint(exit_maps: List, mass_norm: bool = True) -> List[np.ndarray]:
    """
    Joint scaling across exits (per sample): min/max computed over all exits,
    then optional L1 mass normalization for size-invariance of deltas.
    """
    mats = [_to_hw(m) for m in exit_maps]
    mmin = min(float(m.min()) for m in mats)
    mmax = max(float(m.max()) for m in mats)
    rng = max(mmax - mmin, 1e-8)
    mats = [(m - mmin) / rng for m in mats]
    if mass_norm:
        mats = [m / (m.sum() + 1e-8) for m in mats]
    return mats

def _select_maps(exit_maps: List, normalize_mode: str = "joint", mass_norm: bool = True) -> List[np.ndarray]:
    if normalize_mode == "joint":
        return _prep_maps_joint(exit_maps, mass_norm=mass_norm)
    elif normalize_mode == "permap":
        return _prep_maps_permap(exit_maps)
    else:
        raise ValueError("normalize_mode must be 'joint' or 'permap'")

def _iou_topk(a: np.ndarray, b: np.ndarray, k: float = 0.2) -> float:
    """IoU between top-k% pixels of two maps (thresholded by their own percentiles)."""
    ta, tb = np.percentile(a, 100 * (1 - k)), np.percentile(b, 100 * (1 - k))
    ma, mb = a >= ta, b >= tb
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum() + 1e-8
    return float(inter / union)

def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    v1, v2 = a.ravel(), b.ravel()
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
    return float(1.0 - np.dot(v1, v2) / denom)


# ==============================
# Core semantic metrics you KEEP
# ==============================
def mfa_monotonicity(exit_maps: List, normalize_mode="joint", mass_norm=True) -> int:
    """
    1 if similarity-to-final (cosine) is non-decreasing with exit index, else 0.
    """
    A = _select_maps(exit_maps, normalize_mode, mass_norm)
    if len(A) < 3: return 0
    ref = A[-1]
    sims = [1.0 - _cos_dist(a, ref) for a in A[:-1]]
    return int(all(sims[i] <= sims[i + 1] for i in range(len(sims) - 1)))

def IoU_vs_final(exit_maps: Iterable, k: float = 0.2, normalize_mode="joint", mass_norm=True) -> float:
    """
    Mean IoU( A_i , A_L ) across exits i=1..L-1.
    Uses per-map percentiles for both maps (reviewer-standard).
    """
    A = _select_maps(exit_maps, normalize_mode, mass_norm)
    if len(A) < 2: return float("nan")
    ref = A[-1]
    ious = []
    for i in range(len(A) - 1):
        ious.append(_iou_topk(A[i], ref, k=k))
    return float(np.mean(ious))


# ==============================
# Faithfulness (normalized Deletion/Insertion AUC + optional AOPC)
# ==============================
def _preprocess_map_for_mask(m_hw: np.ndarray, relu: bool = True, gaussian_ksize: int = 3) -> np.ndarray:
    m = m_hw.copy()
    if relu:
        m = np.maximum(m, 0.0)
    if cv2 is not None and gaussian_ksize and gaussian_ksize >= 3:
        k = gaussian_ksize | 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    m = _minmax01(m)
    return m

def _upsample_to_input(m_hw: np.ndarray, x: torch.Tensor) -> torch.Tensor:
    m = torch.from_numpy(m_hw).float()[None, None]
    H0, W0 = x.shape[-2:]
    return F.interpolate(m, size=(H0, W0), mode="bilinear", align_corners=False).clamp_(0, 1)

def _box_blur(x: torch.Tensor, k: int = 11) -> torch.Tensor:
    k = max(3, k | 1)
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

def _baseline_image(x: torch.Tensor, kind: str = "mean", blur_k: int = 15) -> torch.Tensor:
    if kind == "blur":
        return _box_blur(x, k=blur_k)
    if kind == "black":
        return torch.zeros_like(x)
    if kind == "mean":
        mean_val = x.mean(dim=(2, 3), keepdim=True)
        return mean_val.expand_as(x)
    raise ValueError("baseline kind must be 'mean'|'blur'|'black'")

@torch.no_grad()
def _final_logits(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, (list, tuple)):
        for cand in reversed(out):
            if torch.is_tensor(cand) and cand.dim() == 2:
                return cand
        for cand in reversed(out):
            if torch.is_tensor(cand): return cand.reshape(cand.size(0), -1)
    elif torch.is_tensor(out):
        return out
    raise RuntimeError("Could not extract final logits")

@torch.no_grad()
def _final_probs(model, x: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    logits = _final_logits(model, x)
    if T != 1.0:
        logits = logits / T
    return torch.softmax(logits, dim=1)

@torch.no_grad()
def _score_prob_for_class(model, x: torch.Tensor, cls: int, T: float = 1.0) -> float:
    probs = _final_probs(model, x, T=T)
    return float(probs[0, cls].item())

def _superpixels(mask_hw: np.ndarray, n_segments: int = 200, compactness: float = 10.0) -> np.ndarray:
    """Return segment labels [H,W] using SLIC; fallback: grid if skimage missing."""
    H, W = mask_hw.shape
    if _USE_SKIMAGE:
        arr = np.stack([mask_hw] * 3, axis=-1).astype(np.float32)
        try:
            seg = slic(arr, n_segments=n_segments, compactness=compactness, start_label=0)
            return seg.astype(np.int32)
        except Exception:
            pass
    # grid fallback
    g = int(max(4, np.sqrt(n_segments)))
    hs, ws = max(1, H // g), max(1, W // g)
    seg = np.zeros((H, W), dtype=np.int32)
    idx = 0
    for i in range(0, H, hs):
        for j in range(0, W, ws):
            seg[i:i + hs, j:j + ws] = idx
            idx += 1
    return seg

def _rank_segments(mask_hw: np.ndarray, seg: np.ndarray) -> List[int]:
    K = int(seg.max()) + 1
    sums = np.bincount(seg.reshape(-1), weights=mask_hw.reshape(-1), minlength=K)
    counts = np.bincount(seg.reshape(-1), minlength=K) + 1e-8
    means = sums / counts
    return list(np.argsort(means)[::-1])

def _apply_fraction_mask_superpixel(x: torch.Tensor, seg: np.ndarray, order: List[int],
                                    frac: float, mode: str, base: torch.Tensor):
    B, C, H, W = x.shape
    target = int(frac * H * W)
    chosen = []
    counts = np.bincount(seg.reshape(-1), minlength=int(seg.max()) + 1)
    covered = 0
    for sid in order:
        chosen.append(sid)
        covered += int(counts[sid])
        if covered >= target:
            break
    mask_np = np.isin(seg, np.array(chosen, dtype=np.int32)).astype(np.float32)
    mask = torch.from_numpy(mask_np).to(x.device).view(1, 1, H, W)
    if mode == "deletion":
        return x * (1 - mask) + base * mask
    else:
        return base * (1 - mask) + x * mask

@torch.no_grad()
def faithfulness_auc_norm(model, x: torch.Tensor, map_hw: np.ndarray, cls: int,
                          steps: int = 40, mode: str = "deletion",
                          T: float = 2.0, baseline: str = "mean",
                          relu: bool = True, gaussian_ksize: int = 3,
                          superpixel: bool = True, n_segments: int = 200, compactness: float = 10.0):
    """
    Normalized AUC of the probability curve s_hat ∈ [0,1].
    Deletion: lower is better; Insertion: higher is better.
    """
    # preprocess and upsample mask
    def _preprocess_map_for_mask(m_hw: np.ndarray, relu=True, gaussian_ksize=3) -> np.ndarray:
        m = m_hw.copy()
        if relu: m = np.maximum(m, 0.0)
        if cv2 is not None and gaussian_ksize and gaussian_ksize >= 3:
            k = gaussian_ksize | 1
            m = cv2.GaussianBlur(m, (k, k), 0)
        return _minmax01(m)

    def _upsample_to_input(m_hw: np.ndarray, x: torch.Tensor) -> torch.Tensor:
        m = torch.from_numpy(m_hw).float()[None, None]
        H0, W0 = x.shape[-2:]
        return F.interpolate(m, size=(H0, W0), mode="bilinear", align_corners=False).clamp_(0, 1)

    m = _preprocess_map_for_mask(map_hw, relu=relu, gaussian_ksize=gaussian_ksize)
    m_up = _upsample_to_input(m, x).to(x.device).squeeze().cpu().numpy()

    # baseline + normalization
    def _baseline_image(x: torch.Tensor, kind: str = "mean", blur_k: int = 15) -> torch.Tensor:
        if kind == "blur": return F.avg_pool2d(x, kernel_size=blur_k, stride=1, padding=blur_k // 2)
        if kind == "black": return torch.zeros_like(x)
        if kind == "mean":
            mean_val = x.mean(dim=(2, 3), keepdim=True)
            return mean_val.expand_as(x)
        raise ValueError

    base = _baseline_image(x, kind=baseline, blur_k=15)
    s0 = _score_prob_for_class(model, x, cls, T=T)
    sb = _score_prob_for_class(model, base, cls, T=T)
    denom = max(s0 - sb, 1e-8)

    fracs = torch.linspace(0, 1, steps + 1, device=x.device)
    vals = []

    # superpixel masking (preferred)
    seg = _superpixels(m_up, n_segments=n_segments, compactness=compactness) if superpixel else None
    if superpixel:
        order_seg = _rank_segments(m_up, seg)
        for f in fracs:
            x_mod = _apply_fraction_mask_superpixel(x, seg, order_seg, float(f.item()), mode, base)
            s = _score_prob_for_class(model, x_mod, cls, T=T)
            vals.append(s)
    else:
        flat = torch.from_numpy(m_up.astype(np.float32)).flatten()
        order = torch.argsort(flat, descending=True).to(x.device)
        H0, W0 = x.shape[-2:]
        for f in fracs:
            k = int(float(f.item()) * H0 * W0)
            mask = torch.zeros((H0 * W0,), dtype=torch.float32, device=x.device)
            mask[order[:k]] = 1.0
            mask = mask.view(1, 1, H0, W0)
            if mode == "deletion":
                x_mod = x * (1 - mask) + base * mask
            else:
                x_mod = base * (1 - mask) + x * mask
            s = _score_prob_for_class(model, x_mod, cls, T=T)
            vals.append(s)

    s = torch.tensor(vals, device=x.device)
    s_hat = torch.clamp((s - sb) / denom, 0.0, 1.0)
    auc = torch.trapz(s_hat, fracs).item()
    return float(auc)

def _per_sample_faithfulness(model, x: torch.Tensor, exit_maps: list, cls: int,
                             steps: int = 40, T: float = 2.0, baseline: str = "mean",
                             relu: bool = True, gaussian_ksize: int = 3,
                             superpixel: bool = True, n_segments: int = 200, compactness: float = 10.0):
    maps_hw = _prep_maps_permap(exit_maps)  # faithfulness uses per-map scaling
    dels, ins = [], []
    for m_hw in maps_hw:
        dels.append(faithfulness_auc_norm(
            model, x, m_hw, cls, steps=steps, mode="deletion",
            T=T, baseline=baseline, relu=relu, gaussian_ksize=gaussian_ksize,
            superpixel=superpixel, n_segments=n_segments, compactness=compactness
        ))
        ins.append(faithfulness_auc_norm(
            model, x, m_hw, cls, steps=steps, mode="insertion",
            T=T, baseline=baseline, relu=relu, gaussian_ksize=gaussian_ksize,
            superpixel=superpixel, n_segments=n_segments, compactness=compactness
        ))
    return {
        "deletion_auc_mean": float(np.mean(dels)) if dels else float("nan"),
        "insertion_auc_mean": float(np.mean(ins)) if ins else float("nan"),
    }


# ==============================
# Running aggregator
# ==============================
class Running:
    def __init__(self): self.v = []
    def add(self, x):
        if np.isfinite(x): self.v.append(float(x))
    def mean_std(self):
        if not self.v: return float("nan"), float("nan")
        a = np.array(self.v, dtype=np.float64)
        return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0)


# ==============================
# Unified evaluator — FINAL CORE SET ONLY
# ==============================
def evaluate_core(model, testloader, device,
                  max_samples: int = 100,
                  include_pfam: bool = True,
                  # Semantic settings
                  topk: float = 0.20,
                  normalize_mode: str = "joint",  # 'joint' recommended
                  mass_norm: bool = True,
                  # Faithfulness settings
                  steps: int = 40, baseline: str = "mean",
                  relu: bool = True, gaussian_ksize: int = 3,
                  superpixel: bool = True, n_segments: int = 200, compactness: float = 10.0,
                  temp: float = 2.0,
                  faithfulness_only_on_correct: bool = True,
                  # Attribution collection
                  force_full: bool = True):
    """
    Computes ONLY the Final Core Set:
      - MFA monotonicity (↑)
      - Convergence IoU vs final (↑)
      - Faithfulness AUCs: Deletion (↓) and Insertion (↑)

    Force full-path attributions for fair evaluation.
    """
    model.eval()
    keys = ["MFA", "IOU_VF", "DEL", "INS"]
    stats: Dict[str, Dict[str, Running]] = {k: defaultdict(Running) for k in keys}

    seen = 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        for i in range(inputs.size(0)):
            if seen >= max_samples: break
            seen += 1
            x = inputs[i:i + 1]
            y = int(labels[i].item())

            # Attribution collection (prefer qty which should already be full-path)
            all_attributions = getAttribution_qty(model, x)
            all_attributions = reorganize_attribution_maps(all_attributions)
            if include_pfam and "PFAM" not in all_attributions:
                # If your collector returns PFAM separately, you can inject here
                pass

            # choose class for faithfulness (GT if correct, else predicted)
            probs = _final_probs(model, x, T=temp)
            pred_cls = int(torch.argmax(probs, dim=1).item())
            cls = y if pred_cls == y else pred_cls
            faith_ok = (pred_cls == y) if faithfulness_only_on_correct else True

            for name, exit_maps in all_attributions.items():
                if not isinstance(exit_maps, list) or len(exit_maps) < 2:
                    continue

                # Semantic metrics
                stats["MFA"][name].add(mfa_monotonicity(exit_maps, normalize_mode, mass_norm))
                stats["IOU_VF"][name].add(IoU_vs_final(exit_maps, k=topk, normalize_mode=normalize_mode, mass_norm=mass_norm))

                # Faithfulness (mean over exits)
                if faith_ok:
                    res = _per_sample_faithfulness(
                        model, x, exit_maps, cls,
                        steps=steps, T=temp, baseline=baseline,
                        relu=relu, gaussian_ksize=gaussian_ksize,
                        superpixel=superpixel, n_segments=n_segments, compactness=compactness
                    )
                    stats["DEL"][name].add(res["deletion_auc_mean"])
                    stats["INS"][name].add(res["insertion_auc_mean"])

        if seen >= max_samples: break

    # Pretty-print
    def _print_block(title, key, note):
        print(f"\n{title} on first {seen} samples ({note})")
        print("{:<14s}  {:>8s}  {:>8s}".format("Method", "Mean", "Std"))
        print("-" * 36)
        for m, rs in stats[key].items():
            mu, sd = rs.mean_std()
            print("{:<14s}  {:>8.3f}  {:>8.3f}".format(m, mu, sd))

    _print_block("MFA monotonicity ↑", "MFA", "higher = better")
    _print_block(f"Convergence IoU vs final (avg, top-{int(100*topk)}%) ↑", "IOU_VF", "higher = better")
    _print_block("Faithfulness (normalized): Deletion AUC ↓", "DEL", "lower = better (0..1)")
    _print_block("Faithfulness (normalized): Insertion AUC ↑", "INS", "higher = better (0..1)")

    # Config echo
    print("\n[Config] normalize_mode =", normalize_mode, "| mass_norm =", mass_norm,
          "| baseline =", baseline, "| superpixel =", superpixel,
          "| n_segments =", n_segments, "| temp =", temp,
          "| faithfulness_only_on_correct =", faithfulness_only_on_correct,
          "| force_full =", force_full)

    return stats
