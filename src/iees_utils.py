import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_COMMON_RESOLUTION = (224, 224)
def generate_gradcam(activations, gradients, common_resolution=DEFAULT_COMMON_RESOLUTION):
    """
    Grad-CAM attribution map for a single exit  (paper Eq. 2).

    Parameters
    ----------
    activations : torch.Tensor  shape (1, K, H, W)
        Feature maps Z_{i,k}(x) at exit i.
    gradients : torch.Tensor  shape (1, K, H, W)
        Gradients of the target class score w.r.t. activations.
    common_resolution : tuple (H0, W0)
        All maps are upsampled to this resolution before PFAM aggregation
        (paper Eq. 3).  Defaults to DEFAULT_COMMON_RESOLUTION.

    Returns
    -------
    cam : np.ndarray  shape (H0, W0)   — min-max normalised to [0, 1]
    channel_weighted_map : np.ndarray  shape (H0, W0)
        Raw (pre-ReLU, pre-norm) channel-weighted map M_i(x) needed for
        computing attribution strength F_i(x)  (paper Eq. 6).
    """
    # alpha_{i,k} = mean over spatial dims of gradient  (paper Eq. 2)
    alpha = gradients.mean(dim=(2, 3), keepdim=True)          # (1, K, 1, 1)

    # Channel-weighted activation map M_i(x)  (paper Eq. 6)
    # M_i(x)[h,w] = sum_k  alpha_{i,k} * Z_{i,k}(x)[h,w]
    M = (alpha * activations).sum(dim=1, keepdim=True)        # (1, 1, H, W)

    # Upsample M to common resolution before storing / using for F_i
    M_up = F.interpolate(
        M, size=common_resolution, mode='bilinear', align_corners=False
    )
    channel_weighted_map = M_up.detach().squeeze().cpu().numpy()  # (H0, W0)

    # Grad-CAM  A_i(x) = ReLU(M_i(x))  then normalise
    cam_raw = torch.relu(M_up)
    cam = cam_raw.detach().squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)

    return cam, channel_weighted_map

def generate_pfams(activations, gradients, input_tensor, pfams_list,
                   channel_weighted_maps_list,
                   common_resolution=DEFAULT_COMMON_RESOLUTION):
    """
    Generate exit-wise Grad-CAM map, append to pfams_list, and compute the
    cumulative PFAM map C_i(x)  (paper Eq. 3).

    Parameters
    ----------
    activations : torch.Tensor  shape (1, K, H, W)
    gradients   : torch.Tensor  shape (1, K, H, W)
    input_tensor : torch.Tensor   (used only to infer common_resolution if
                                    the caller passes None)
    pfams_list  : list of np.ndarray
        Accumulated exit-wise maps  A_1 … A_{i-1}.  Updated in-place.
    channel_weighted_maps_list : list of np.ndarray
        Accumulated channel-weighted maps M_1 … M_{i-1}.  Updated in-place.
        Required later by compute_attribution_strength (F_i).
    common_resolution : tuple (H0, W0)

    Returns
    -------
    pfams_list             : list — updated with A_i
    cumulative_map         : np.ndarray  shape (H0, W0) — C_i(x)
    channel_weighted_maps_list : list — updated with M_i
    """
    cam, channel_weighted_map = generate_gradcam(
        activations, gradients, common_resolution=common_resolution
    )

    pfams_list.append(cam)
    channel_weighted_maps_list.append(channel_weighted_map)

    # C_i(x) = (1/i) * sum_{j=1}^{i} A_j(x)   paper Eq. (3)
    # np.mean over axis-0 of a stack is identical to (1/i)*sum
    cumulative_map = np.mean(np.stack(pfams_list), axis=0)

    return pfams_list, cumulative_map, channel_weighted_maps_list


def calculate_consistency_index(cumulative_maps):
    """
    Compute attribution stability Q_i(x)  (paper Eq. 4).

    Q_i(x) = mean_{p} |C_i(x)[p] - C_{i-1}(x)[p]|

    Convention: C_0(x) = 0  →  Q_1(x) = 0  (paper text after Eq. 4).
    Smaller Q_i means more stable attribution focus (lower is better).

    Parameters
    ----------
    cumulative_maps : list of np.ndarray
        C_1(x), C_2(x), …, C_i(x) — cumulative maps in exit order.

    Returns
    -------
    float : Q_i(x).  Returns 0.0 for the very first exit (i=1).
    """
    if len(cumulative_maps) < 2:
        # i = 1:  C_0 := 0  →  Q_1 = mean|C_1 - 0| — paper sets Q_1 = 0
        return 0.0

    Ci      = cumulative_maps[-1]   # C_i(x)
    Ci_prev = cumulative_maps[-2]   # C_{i-1}(x)

    # mean absolute difference over all spatial locations Ω
    Q_i = np.mean(np.abs(Ci - Ci_prev))
    return float(Q_i)



def compute_attribution_strength(channel_weighted_map, k=20):
    """
    Compute attribution strength F_i(x) from the channel-weighted map M_i(x).

    Uses the top-k% most discriminative spatial locations (paper Eq. 7, 8).
    Default k=20 as stated in paper Section 3.3 and Appendix P.

    Parameters
    ----------
    channel_weighted_map : np.ndarray  shape (H0, W0)
        M_i(x) — raw channel-weighted activation map (NOT ReLU'd, NOT normalised).
        Produced by generate_gradcam as the second return value.
    k : int
        Percentage of top salient spatial locations to retain (default 20).

    Returns
    -------
    float : F_i(x) — mean activation over the top-k% salient mask.
            Positive values → strong, spatially concentrated class evidence.
            Near-zero / negative → weak or diffuse support.
    """
    # Quantile threshold  q_k^i(x) = (1 - k/100)-quantile of M_i(x)  (Eq. 8)
    q = np.quantile(channel_weighted_map, 1.0 - k / 100.0)

    # Salient mask  Omega_k^i(x)
    mask = channel_weighted_map >= q                          # bool (H0, W0)

    if mask.sum() == 0:
        return 0.0

    # F_i(x) = average of M_i(x) over the salient mask  (Eq. 7)
    F_i = channel_weighted_map[mask].mean()
    return float(F_i)



def minmax_normalize(values, eps=1e-8):
    """
    Min-max normalise a list/array of scalar values to [0, 1].

    Parameters
    ----------
    values : list or np.ndarray of floats
    eps    : small constant for numerical stability

    Returns
    -------
    np.ndarray  shape (N,)  normalised to [0, 1]
    """
    arr = np.array(values, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + eps)



def compute_iees_score(activations, gradients, output, consistency_index, w,
                       k=20, common_resolution=DEFAULT_COMMON_RESOLUTION):
    """
    Compute the oracle IEES score for one exit  (paper Eq. 5).

    IEES_i(x) = w1 * S_i(x)  +  w2 * F_i(x)  -  w3 * Q_i(x)

        S_i  : max softmax confidence                         (+, rewarded)
        F_i  : attribution strength over top-k% salient mask (+, rewarded)
        Q_i  : attribution stability (mean |C_i - C_{i-1}|)  (-, penalised)

    F_i and Q_i must be min-max normalised to [0,1] across the validation set
    before scoring (paper Appendix M).  When called per-sample at calibration
    time, pass already-normalised values for F and Q, or normalise the
    returned values externally.

    Parameters
    ----------
    activations       : torch.Tensor  (1, K, H, W)
    gradients         : torch.Tensor  (1, K, H, W)
    output            : torch.Tensor  (1, C)  — logits at exit i
    consistency_index : float  — Q_i(x), already computed via
                        calculate_consistency_index (paper Eq. 4).
                        Should be min-max normalised before passing.
    w                 : list [w1, w2, w3]  — non-negative weights
                        (tuned on validation; paper Sec 3.3, Appendix M)
    k                 : int  — top-k% mask for F_i  (default 20, Appendix P)
    common_resolution : tuple  — for generate_gradcam upsampling

    Returns
    -------
    iees_score  : float  — oracle IEES_i(x)
    S_iees      : float  — softmax confidence  S_i(x)
    F_iees      : float  — attribution strength  F_i(x)
    Q_iees      : float  — stability index  Q_i(x)  (= consistency_index)
    activation_score : float  — mean activation (diagnostic)
    gradient_score   : float  — mean |gradient| (diagnostic)
    """
    w1, w2, w3 = w[0], w[1], w[2]

    # --- S_i(x): max softmax confidence  (paper Eq. 5, S_i = max_c p_i(c|x)) ---
    S_iees = F.softmax(output, dim=1).max(dim=1).values.item()

    # --- M_i(x) and F_i(x): attribution strength  (paper Eq. 6, 7, 8) ---
    _, channel_weighted_map = generate_gradcam(
        activations, gradients, common_resolution=common_resolution
    )
    F_iees = compute_attribution_strength(channel_weighted_map, k=k)

    # --- Q_i(x): stability (passed in, computed by calculate_consistency_index) ---
    Q_iees = float(consistency_index)

    # --- IEES_i(x) = w1*S + w2*F - w3*Q  (paper Eq. 5) ---
    # Sign: confidence (+), attribution strength (+), instability (-)
    iees_score = w1 * S_iees + w2 * F_iees - w3 * Q_iees

    # Diagnostic scalars (kept for backward compatibility)
    activation_score = activations.mean().item()
    gradient_score   = gradients.abs().mean().item()

    return iees_score, S_iees, F_iees, Q_iees, activation_score, gradient_score

def normalize_tensor(tensor, eps=1e-6):
    """
    Z-score normalise a tensor (zero mean, unit variance).
    Kept for backward compatibility; not used in core IEES computation.
    """
    return (tensor - tensor.mean()) / (tensor.std() + eps)
