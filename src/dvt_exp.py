# src/dvt_experiment_iees_pfam_synced.py
# DVT T2T-ViT-12 experiment for rebuttal R3-Q2
#
# Synced with updated IEES/PFAM utilities:
#   - PFAM: Grad-CAM maps are upsampled to a common resolution before accumulation.
#   - F_i: top-k% attribution strength from the raw pre-ReLU channel-weighted map M_i.
#   - Q_i: mean absolute difference between consecutive cumulative PFAM maps.
#   - IEES_i = w1*S_i + w2*F_i - w3*Q_i.
#   - Oracle IEES/PFAM is computed offline for calibration/validation targets only.
#   - Inference uses an attribution-free RF proxy; no Grad-CAM/PFAM backward pass at test time.

import sys
import os
import json
import time
import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ── Add DVT to path ──────────────────────────────────────────────────────────
DVT_PATH = os.path.join(os.path.dirname(__file__), 'dvt')
if DVT_PATH not in sys.path:
    sys.path.insert(0, DVT_PATH)

from timm.models import create_model  # type: ignore
import models  # noqa: F401  # type: ignore  # DVT registration side-effects

# ── Config ───────────────────────────────────────────────────────────────────
IMAGENET_DIR = os.environ.get('IMAGENET_DIR', '../data/imagenet')
CHECKPOINT = os.environ.get('DVT_CHECKPOINT', '../models/DVT_T2t_vit_12.pth.tar')
RESULTS_DIR = '../results'
FEATURES_DIR = '../results/dvt_iees_pfam_synced_features'
CHECKPOINT_FILE = '../results/dvt_iees_pfam_synced_checkpoint.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EXITS = 3
TOKEN_GRIDS = [7, 10, 14]
COMMON_RESOLUTION = (224, 224)
TOP_K_PERCENT = 20

# Use TEST MODE values while debugging. For final paper run, restore large values.
BATCH_SIZE_CAL = 16          # extraction still processes one image at a time internally
BATCH_SIZE_EVAL = 1          # required for per-sample early-exit decisions
CAL_SIZE = 500               # set to 25000 for final run
VAL_SIZE = 200               # set to 12000 for final run
TEST_SIZE = 200              # set to 12000 for final run
SEED = 42

IEES_WEIGHTS = (0.5, 0.3, 0.2)

print('⚠️  TEST MODE: edit CAL_SIZE/VAL_SIZE/TEST_SIZE for the final paper run.')
print(f'Device: {DEVICE}')
print(f'Exits: {NUM_EXITS} | token grids: {TOKEN_GRIDS}')


# ── Checkpoint helpers ───────────────────────────────────────────────────────
def save_checkpoint(data: Dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data['last_updated'] = datetime.datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print('  💾 Checkpoint saved!')


def load_checkpoint() -> Optional[Dict]:
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE, encoding='utf-8') as f:
        data = json.load(f)
    print(f"  ⚡ Checkpoint found! Done: {data.get('completed_steps', [])}")
    return data


def checkpoint_has(ckpt: Optional[Dict], step: str) -> bool:
    return ckpt is not None and step in ckpt.get('completed_steps', [])


def checkpoint_mark(ckpt: Optional[Dict], step: str) -> Dict:
    if ckpt is None:
        ckpt = {'completed_steps': []}
    ckpt.setdefault('completed_steps', [])
    if step not in ckpt['completed_steps']:
        ckpt['completed_steps'].append(step)
    return ckpt


# ── Feature save/load ────────────────────────────────────────────────────────
def save_raw_features(split_name: str,
                      X: np.ndarray,
                      S: np.ndarray,
                      F_raw: np.ndarray,
                      Q_raw: np.ndarray,
                      y_corr: np.ndarray,
                      exits: np.ndarray) -> None:
    os.makedirs(FEATURES_DIR, exist_ok=True)
    prefix = os.path.join(FEATURES_DIR, split_name)
    np.save(f'{prefix}_X.npy', X)
    np.save(f'{prefix}_S.npy', S)
    np.save(f'{prefix}_F_raw.npy', F_raw)
    np.save(f'{prefix}_Q_raw.npy', Q_raw)
    np.save(f'{prefix}_y_corr.npy', y_corr)
    np.save(f'{prefix}_exits.npy', exits)
    mb = (X.nbytes + S.nbytes + F_raw.nbytes + Q_raw.nbytes + y_corr.nbytes + exits.nbytes) / 1024 / 1024
    print(f'  ✅ Saved {split_name}: {len(X)} exit-samples ({mb:.1f} MB)')


def load_raw_features(split_name: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    prefix = os.path.join(FEATURES_DIR, split_name)
    files = [
        f'{prefix}_X.npy', f'{prefix}_S.npy', f'{prefix}_F_raw.npy',
        f'{prefix}_Q_raw.npy', f'{prefix}_y_corr.npy', f'{prefix}_exits.npy'
    ]
    if not all(os.path.exists(f) for f in files):
        return None
    X = np.load(f'{prefix}_X.npy')
    S = np.load(f'{prefix}_S.npy')
    F_raw = np.load(f'{prefix}_F_raw.npy')
    Q_raw = np.load(f'{prefix}_Q_raw.npy')
    y_corr = np.load(f'{prefix}_y_corr.npy')
    exits = np.load(f'{prefix}_exits.npy')
    print(f'  ✅ Loaded {split_name}: {len(X)} exit-samples')
    return X, S, F_raw, Q_raw, y_corr, exits


def raw_features_exist(split_name: str) -> bool:
    return os.path.exists(os.path.join(FEATURES_DIR, f'{split_name}_X.npy'))


# ── Updated IEES/PFAM utilities, mirrored from iees_utils.py/pfam_utils.py ───
def minmax_normalize_with_params(values: np.ndarray,
                                 lo: Optional[float] = None,
                                 hi: Optional[float] = None,
                                 eps: float = 1e-8) -> Tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if lo is None:
        lo = float(arr.min())
    if hi is None:
        hi = float(arr.max())
    return ((arr - lo) / (hi - lo + eps)).astype(np.float32), lo, hi


def generate_gradcam(activations: torch.Tensor,
                     gradients: torch.Tensor,
                     common_resolution: Tuple[int, int] = COMMON_RESOLUTION) -> Tuple[np.ndarray, np.ndarray]:
    """Grad-CAM map A_i and raw pre-ReLU channel-weighted map M_i."""
    alpha = gradients.mean(dim=(2, 3), keepdim=True)
    M = (alpha * activations).sum(dim=1, keepdim=True)

    M_up = F.interpolate(M, size=common_resolution, mode='bilinear', align_corners=False)
    channel_weighted_map = M_up.detach().squeeze().cpu().numpy()

    cam_raw = torch.relu(M_up)
    cam = cam_raw.detach().squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
    return cam, channel_weighted_map


def generate_pfams(activations: torch.Tensor,
                   gradients: torch.Tensor,
                   pfams_list: List[np.ndarray],
                   channel_weighted_maps_list: List[np.ndarray],
                   common_resolution: Tuple[int, int] = COMMON_RESOLUTION) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
    cam, channel_weighted_map = generate_gradcam(activations, gradients, common_resolution)
    pfams_list.append(cam)
    channel_weighted_maps_list.append(channel_weighted_map)
    cumulative_map = np.mean(np.stack(pfams_list), axis=0)
    return pfams_list, cumulative_map, channel_weighted_maps_list


def calculate_consistency_index(cumulative_maps: List[np.ndarray]) -> float:
    """Q_i = mean_p |C_i[p] - C_{i-1}[p]|. Q_1 is set to 0."""
    if len(cumulative_maps) < 2:
        return 0.0
    return float(np.mean(np.abs(cumulative_maps[-1] - cumulative_maps[-2])))


def compute_attribution_strength(channel_weighted_map: np.ndarray, k: int = TOP_K_PERCENT) -> float:
    q = np.quantile(channel_weighted_map, 1.0 - k / 100.0)
    mask = channel_weighted_map >= q
    if mask.sum() == 0:
        return 0.0
    return float(channel_weighted_map[mask].mean())


def compute_iees_from_components(S: np.ndarray,
                                 F_norm: np.ndarray,
                                 Q_norm: np.ndarray,
                                 weights: Tuple[float, float, float]) -> np.ndarray:
    w1, w2, w3 = weights
    return (w1 * S + w2 * F_norm - w3 * Q_norm).astype(np.float32)


# ── Data ─────────────────────────────────────────────────────────────────────
def build_label_remap(dataset: datasets.ImageFolder) -> Dict[int, int]:
    standard = list(ResNet18_Weights.IMAGENET1K_V1.meta['categories'])

    def norm(s: str) -> str:
        return s.lower().replace(' ', '_').replace('-', '_').replace("'", '_').replace(',', '')

    std_lookup = {norm(name): idx for idx, name in enumerate(standard)}
    remap: Dict[int, int] = {}
    not_found: List[str] = []
    for your_idx, cls in enumerate(dataset.classes):
        normed = norm(cls)
        if normed in std_lookup:
            remap[your_idx] = std_lookup[normed]
        else:
            not_found.append(cls)
            remap[your_idx] = your_idx

    matched = len(remap) - len(not_found)
    print(f'  Label remap: {matched}/{len(remap)} matched')
    if not_found:
        print(f'  ⚠️ Unmatched classes (first 10): {not_found[:10]}')
    return remap


def load_imagenet_splits(data_dir: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    print('Loading ImageNet...')
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f'  Total: {len(dataset)}')

    remap = build_label_remap(dataset)
    dataset.targets = [remap[t] for t in dataset.targets]
    dataset.samples = [(p, remap[l]) for p, l in dataset.samples]

    total = CAL_SIZE + VAL_SIZE + TEST_SIZE
    if total > len(dataset):
        raise ValueError(f'Requested {total} samples but dataset has only {len(dataset)}.')

    torch.manual_seed(SEED)
    indices = torch.randperm(len(dataset))[:total].tolist()
    cal_set = Subset(dataset, indices[:CAL_SIZE])
    val_set = Subset(dataset, indices[CAL_SIZE:CAL_SIZE + VAL_SIZE])
    test_set = Subset(dataset, indices[CAL_SIZE + VAL_SIZE:])

    print(f'  Cal: {len(cal_set)}, Val: {len(val_set)}, Test: {len(test_set)}')

    cal_loader = DataLoader(cal_set, batch_size=BATCH_SIZE_CAL, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=0)
    return cal_loader, val_loader, test_loader


# ── DVT model and ViT GradCAM hook ───────────────────────────────────────────
def load_dvt_model(checkpoint_path: str, device: torch.device):
    print('Building DVT T2T-ViT-12...')
    model = create_model(
        'DVT_T2t_vit_12', pretrained=False, num_classes=1000,
        drop_rate=0.0, drop_connect_rate=None, drop_path_rate=0.1,
        drop_block_rate=None, global_pool=None, bn_tf=False,
        bn_momentum=None, bn_eps=None, checkpoint_path='',
    )

    print('Loading pretrained weights...')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict)
    reported_acc = ckpt.get('anytime_classification', None)

    model.num_exits = NUM_EXITS
    model.activations = None
    model.gradients = None
    model = model.to(device)
    model.eval()

    print('✅ DVT loaded!')
    if reported_acc is not None:
        print(f'   Official per-exit acc: {reported_acc}')
    return model, reported_acc


def reshape_vit_tokens(tensor: torch.Tensor, H: int, W: int, remove_cls: bool = True) -> torch.Tensor:
    B, N, D = tensor.shape
    expected = H * W
    if remove_cls and N == expected + 1:
        tensor = tensor[:, 1:, :]
        N = expected
    elif N != expected:
        raise ValueError(f'Cannot reshape tokens: got N={N}, expected {expected} or {expected + 1}.')
    return tensor.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()


def forward_to_exit_dvt(model, x: torch.Tensor, exit_idx: int, gradient_required: bool = False):
    H = W = TOKEN_GRIDS[exit_idx]
    sub_models = [model.less_less_token, model.less_token, model.normal_token]
    sub = sub_models[exit_idx]

    saved_act = [None]
    saved_grad = [None]

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        saved_act[0] = reshape_vit_tokens(out, H, W, remove_cls=True)
        if gradient_required:
            out.retain_grad()

            def save_token_grad(g):
                saved_grad[0] = reshape_vit_tokens(g, H, W, remove_cls=True)

            out.register_hook(save_token_grad)

    handle = sub.blocks[-2].register_forward_hook(hook_fn)
    if gradient_required:
        model.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(gradient_required):
        out1, out2, out3 = model(x)
        logits = [out1, out2, out3][exit_idx]

    handle.remove()
    model.activations = saved_act[0]

    probs = F.softmax(logits, dim=1)
    class_idx = probs.argmax(dim=1)
    target = logits[0, int(class_idx.item())]

    if gradient_required:
        target.backward()

    model.gradients = saved_grad[0]
    if gradient_required and (model.activations is None or model.gradients is None):
        raise RuntimeError(f'Missing activation/gradient at exit {exit_idx + 1}.')

    return logits, class_idx, target


# ── Attribution-free proxy features ──────────────────────────────────────────
def sample_proxy_features_from_logits(logits: torch.Tensor,
                                      prev_state: Optional[dict]) -> Tuple[np.ndarray, dict]:
    probs = F.softmax(logits, dim=1)
    conf = float(probs.max(dim=1).values.item())
    entropy = float(-(probs * torch.log(probs + 1e-8)).sum().item())
    top2 = torch.topk(probs, 2, dim=1).values
    margin = float((top2[:, 0] - top2[:, 1]).item())
    logit_abs_mean = float(logits.abs().mean().item())
    logit_abs_max = float(logits.abs().max().item())
    pred = int(logits.argmax(dim=1).item())

    if prev_state is None:
        conf_delta = 0.0
        class_stability = 0.0
        logit_cos = 0.0
        entropy_delta = 0.0
        act_delta = 0.0
    else:
        conf_delta = abs(conf - prev_state['conf'])
        class_stability = float(pred == prev_state['pred'])
        logit_cos = float(F.cosine_similarity(logits, prev_state['logits'], dim=1).item())
        entropy_delta = abs(entropy - prev_state['entropy'])
        act_delta = abs(logit_abs_mean - prev_state['logit_abs_mean'])

    feat = np.array([
        conf, margin, entropy,
        logit_abs_mean, logit_abs_max,
        conf_delta, class_stability,
        logit_cos, entropy_delta, act_delta,
    ], dtype=np.float32)

    state = {
        'conf': conf,
        'pred': pred,
        'logits': logits.detach(),
        'entropy': entropy,
        'logit_abs_mean': logit_abs_mean,
    }
    return feat, state


# ── Offline oracle extraction ────────────────────────────────────────────────
def extract_oracle_components(model,
                              loader: DataLoader,
                              device: torch.device,
                              desc: str = 'Extracting') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract low-cost proxy features plus oracle IEES components S, F_raw, Q_raw."""
    all_X: List[np.ndarray] = []
    all_S: List[float] = []
    all_F_raw: List[float] = []
    all_Q_raw: List[float] = []
    all_corr: List[float] = []
    all_exits: List[int] = []

    print(f'{desc}: offline oracle PFAM/IEES component extraction...')
    model.eval()
    img_count = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        if batch_idx % 20 == 0:
            print(f'  Batch {batch_idx}/{len(loader)} ({img_count} images done)')

        for i in range(imgs.size(0)):
            img = imgs[i:i + 1].to(device)
            label = labels[i].to(device)
            img_count += 1

            pfams_list: List[np.ndarray] = []
            cumulative_maps: List[np.ndarray] = []
            channel_weighted_maps: List[np.ndarray] = []
            prev_state: Optional[dict] = None

            for exit_idx in range(NUM_EXITS):
                logits, _, _ = forward_to_exit_dvt(model, img, exit_idx, gradient_required=True)
                probs = F.softmax(logits, dim=1)

                pfams_list, cumulative_map, channel_weighted_maps = generate_pfams(
                    model.activations,
                    model.gradients,
                    pfams_list,
                    channel_weighted_maps,
                    common_resolution=COMMON_RESOLUTION,
                )
                cumulative_maps.append(cumulative_map)

                S_i = float(probs.max(dim=1).values.item())
                F_i_raw = compute_attribution_strength(channel_weighted_maps[-1], k=TOP_K_PERCENT)
                Q_i_raw = calculate_consistency_index(cumulative_maps)

                pred = int(probs.argmax(dim=1).item())
                correct = float(pred == int(label.item()))
                feat, prev_state = sample_proxy_features_from_logits(logits, prev_state)

                all_X.append(feat)
                all_S.append(S_i)
                all_F_raw.append(F_i_raw)
                all_Q_raw.append(Q_i_raw)
                all_corr.append(correct)
                all_exits.append(exit_idx)

                model.activations = None
                model.gradients = None

    X = np.asarray(all_X, dtype=np.float32)
    S = np.asarray(all_S, dtype=np.float32)
    F_raw = np.asarray(all_F_raw, dtype=np.float32)
    Q_raw = np.asarray(all_Q_raw, dtype=np.float32)
    y_corr = np.asarray(all_corr, dtype=np.float32)
    exits = np.asarray(all_exits, dtype=np.int32)

    print(f'  Done: {img_count} images × {NUM_EXITS} exits = {len(X)} exit-samples')
    print(f'  F_raw range: [{F_raw.min():.6f}, {F_raw.max():.6f}] | Q_raw range: [{Q_raw.min():.6f}, {Q_raw.max():.6f}]')
    return X, S, F_raw, Q_raw, y_corr, exits


# ── Evaluation policies: no attribution at inference ────────────────────────
def run_no_early_exit(model, loader: DataLoader, device: torch.device):
    correct = 0
    total = 0
    total_time = 0.0
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            t0 = time.perf_counter()
            _, _, out3 = model(imgs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_time += (t1 - t0) * 1000.0
            pred = out3.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))

    acc = 100.0 * correct / total
    dist = [0.0] * NUM_EXITS
    dist[-1] = 100.0
    exit_acc = [0.0] * (NUM_EXITS - 1) + [acc]
    return acc, dist, exit_acc, total_time / total


def run_confidence_policy(model, loader: DataLoader, device: torch.device, threshold: float = 0.9):
    correct = 0
    total = 0
    exit_counts = [0] * NUM_EXITS
    exit_correct = [0] * NUM_EXITS
    total_time = 0.0
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if imgs.size(0) != 1:
                raise ValueError('run_confidence_policy expects batch_size=1 for per-sample exits.')

            t0 = time.perf_counter()
            out1, out2, out3 = model(imgs)
            for exit_idx, logits in enumerate([out1, out2, out3]):
                probs = F.softmax(logits, dim=1)
                conf = float(probs.max(dim=1).values.item())
                if conf >= threshold or exit_idx == NUM_EXITS - 1:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    total_time += (t1 - t0) * 1000.0
                    pred = logits.argmax(dim=1)
                    is_correct = int((pred == labels).item())
                    correct += is_correct
                    total += 1
                    exit_counts[exit_idx] += 1
                    exit_correct[exit_idx] += is_correct
                    break

    acc = 100.0 * correct / total
    dist = [100.0 * c / total for c in exit_counts]
    exit_acc = [100.0 * exit_correct[i] / exit_counts[i] if exit_counts[i] else 0.0 for i in range(NUM_EXITS)]
    return acc, dist, exit_acc, total_time / total


def run_proxy_policy(model, rf, loader: DataLoader, device: torch.device, thresholds: List[float]):
    """IEES proxy policy. No Grad-CAM, no PFAM, no backward pass at inference."""
    correct = 0
    total = 0
    exit_counts = [0] * NUM_EXITS
    exit_correct = [0] * NUM_EXITS
    total_time = 0.0
    model.eval()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if imgs.size(0) != 1:
                raise ValueError('run_proxy_policy expects batch_size=1 for per-sample exits.')

            t0 = time.perf_counter()
            out1, out2, out3 = model(imgs)
            prev_state: Optional[dict] = None

            for exit_idx, logits in enumerate([out1, out2, out3]):
                feat, prev_state = sample_proxy_features_from_logits(logits, prev_state)
                pred_score = float(rf.predict(feat.reshape(1, -1))[0])
                if pred_score >= thresholds[exit_idx] or exit_idx == NUM_EXITS - 1:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    total_time += (t1 - t0) * 1000.0
                    pred = logits.argmax(dim=1)
                    is_correct = int((pred == labels).item())
                    correct += is_correct
                    total += 1
                    exit_counts[exit_idx] += 1
                    exit_correct[exit_idx] += is_correct
                    break

    acc = 100.0 * correct / total
    dist = [100.0 * c / total for c in exit_counts]
    exit_acc = [100.0 * exit_correct[i] / exit_counts[i] if exit_counts[i] else 0.0 for i in range(NUM_EXITS)]
    return acc, dist, exit_acc, total_time / total


# ── Threshold tuning: validation only ────────────────────────────────────────
def tune_conf_threshold(model, val_loader: DataLoader, device: torch.device) -> float:
    print('  Tuning confidence threshold on validation...')
    best_tau = 0.90
    best_acc = -1.0
    for tau in np.arange(0.50, 0.99, 0.05):
        acc, _, _, _ = run_confidence_policy(model, val_loader, device, float(tau))
        if acc > best_acc:
            best_acc = acc
            best_tau = float(tau)
    print(f'  Best confidence tau: {best_tau:.2f} → {best_acc:.2f}%')
    return best_tau


def tune_proxy_thresholds(model, rf, val_loader: DataLoader, device: torch.device) -> List[float]:
    print('  Tuning IEES-proxy thresholds on validation...')
    best_thresholds = [0.7] * NUM_EXITS
    best_acc = -1.0
    # IEES scores may go slightly below/above [0,1] because of signed F before normalization.
    for tau in np.arange(-0.20, 1.05, 0.05):
        thresholds = [float(tau)] * NUM_EXITS
        acc, _, _, _ = run_proxy_policy(model, rf, val_loader, device, thresholds)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = list(thresholds)
    print(f'  Best IEES-proxy tau: {best_thresholds[0]:.2f} → {best_acc:.2f}%')
    return best_thresholds


def find_matched_confidence(model,
                            val_loader: DataLoader,
                            test_loader: DataLoader,
                            device: torch.device,
                            val_iees_dist: List[float]):
    """Choose matched confidence threshold on validation; evaluate once on test."""
    print('  Finding IEES-matched confidence threshold on validation only...')
    best_tau = 0.50
    best_diff = float('inf')
    for tau in np.arange(0.10, 0.99, 0.02):
        _, dist, _, _ = run_confidence_policy(model, val_loader, device, float(tau))
        diff = sum(abs(dist[i] - val_iees_dist[i]) for i in range(NUM_EXITS))
        if diff < best_diff:
            best_diff = diff
            best_tau = float(tau)
    print(f'  Best matched confidence tau: {best_tau:.2f} (validation dist diff={best_diff:.2f}%)')
    return run_confidence_policy(model, test_loader, device, best_tau)


# ── Sanity check and reporting ───────────────────────────────────────────────
def quick_accuracy_check(model, loader: DataLoader, device: torch.device, max_batches: int = 20) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            _, _, out3 = model(imgs)
            pred = out3.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))
    return 100.0 * correct / max(total, 1)


def print_results(results: Dict[str, Tuple], reported_acc=None) -> None:
    print('\n' + '=' * 100)
    print('DVT T2T-ViT-12 / ImageNet — IEES-PFAM-Synced Exit Policy Comparison')
    print('Oracle PFAM/IEES used offline for proxy supervision; test-time IEES uses attribution-free RF proxy.')
    if reported_acc is not None:
        print(f'Official per-exit acc: {reported_acc}')
    print('=' * 100)

    print(f"{'Policy':<28}", end='')
    for i in range(NUM_EXITS):
        print(f'  E{i + 1}% ', end='')
    print('  ', end='')
    for i in range(NUM_EXITS):
        print(f' E{i + 1}Acc ', end='')
    print(f"  {'Overall':>10}  {'Time':>10}")
    print('-' * 100)

    for name, data in results.items():
        acc, dist, exit_acc, t, acc_std = data
        print(f'{name:<28}', end='')
        for d in dist:
            print(f' {d:6.1f}', end='') if d > 0 else print('      –', end='')
        print('  ', end='')
        for a in exit_acc:
            print(f' {a:6.1f}', end='') if a > 0 else print('      –', end='')
        print(f'  {acc:7.2f}±{acc_std:.2f}  {t:7.2f}ms')
    print('=' * 100)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print('\n' + '=' * 70)
    print('DVT T2T-ViT-12 Experiment — synced with updated IEES/PFAM')
    print('=' * 70 + '\n')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    ckpt = load_checkpoint()
    if ckpt is None:
        ckpt = {'completed_steps': []}

    cal_loader, val_loader, test_loader = load_imagenet_splits(IMAGENET_DIR)
    model, reported_acc = load_dvt_model(CHECKPOINT, DEVICE)

    if not checkpoint_has(ckpt, 'acc_check'):
        print('\nQuick accuracy check...')
        quick_acc = quick_accuracy_check(model, val_loader, DEVICE)
        print(f'Quick acc (E3): {quick_acc:.2f}%')
        if quick_acc < 50:
            print('⚠️ Low accuracy. Check ImageNet label mapping / folder names before trusting results.')
            return
        ckpt['quick_acc'] = quick_acc
        ckpt = checkpoint_mark(ckpt, 'acc_check')
        save_checkpoint(ckpt)
    else:
        quick_acc = ckpt.get('quick_acc', 0.0)
        print(f'✅ Acc cached: {quick_acc:.2f}%')

    # Step 1: extract oracle components offline.
    print('\nStep 1: Offline oracle IEES/PFAM component extraction...')
    if raw_features_exist('dvt_cal'):
        loaded = load_raw_features('dvt_cal')
        assert loaded is not None
        X_cal, S_cal, F_cal_raw, Q_cal_raw, y_corr_cal, exits_cal = loaded
    else:
        X_cal, S_cal, F_cal_raw, Q_cal_raw, y_corr_cal, exits_cal = extract_oracle_components(
            model, cal_loader, DEVICE, 'Calibration'
        )
        save_raw_features('dvt_cal', X_cal, S_cal, F_cal_raw, Q_cal_raw, y_corr_cal, exits_cal)
        ckpt = checkpoint_mark(ckpt, 'cal_components')
        save_checkpoint(ckpt)

    if raw_features_exist('dvt_val'):
        loaded = load_raw_features('dvt_val')
        assert loaded is not None
        X_val, S_val, F_val_raw, Q_val_raw, y_corr_val, exits_val = loaded
    else:
        X_val, S_val, F_val_raw, Q_val_raw, y_corr_val, exits_val = extract_oracle_components(
            model, val_loader, DEVICE, 'Validation'
        )
        save_raw_features('dvt_val', X_val, S_val, F_val_raw, Q_val_raw, y_corr_val, exits_val)
        ckpt = checkpoint_mark(ckpt, 'val_components')
        save_checkpoint(ckpt)

    # Step 2: normalize F and Q using calibration statistics, then compute oracle IEES targets.
    print('\nStep 2: Normalizing F/Q and forming oracle IEES targets...')
    F_cal, F_lo, F_hi = minmax_normalize_with_params(F_cal_raw)
    Q_cal, Q_lo, Q_hi = minmax_normalize_with_params(Q_cal_raw)
    F_val, _, _ = minmax_normalize_with_params(F_val_raw, F_lo, F_hi)
    Q_val, _, _ = minmax_normalize_with_params(Q_val_raw, Q_lo, Q_hi)

    y_iees_cal = compute_iees_from_components(S_cal, F_cal, Q_cal, IEES_WEIGHTS)
    y_iees_val = compute_iees_from_components(S_val, F_val, Q_val, IEES_WEIGHTS)

    ckpt['normalization'] = {'F_lo': F_lo, 'F_hi': F_hi, 'Q_lo': Q_lo, 'Q_hi': Q_hi}
    print(f'  F norm range source: [{F_lo:.6f}, {F_hi:.6f}]')
    print(f'  Q norm range source: [{Q_lo:.6f}, {Q_hi:.6f}]')
    print(f'  IEES cal range: [{y_iees_cal.min():.3f}, {y_iees_cal.max():.3f}]')

    # Step 3: Train attribution-free RF proxy.
    print('\nStep 3: Training attribution-free IEES RF proxy...')
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_cal, y_iees_cal)
    y_pred_val = rf.predict(X_val)
    r2 = float(r2_score(y_iees_val, y_pred_val))
    mae = float(mean_absolute_error(y_iees_val, y_pred_val))
    print(f'  R² (val): {r2:.3f} | MAE (val): {mae:.4f}')
    ckpt['r2_val'] = r2
    ckpt['mae_val'] = mae
    ckpt = checkpoint_mark(ckpt, 'rf_iees_trained')
    save_checkpoint(ckpt)

    # Step 4: Tune thresholds on validation only.
    print('\nStep 4: Tuning thresholds on validation...')
    if checkpoint_has(ckpt, 'conf_tau'):
        best_conf_tau = ckpt['best_conf_tau']
        print(f'  ✅ Conf tau: {best_conf_tau:.2f}')
    else:
        best_conf_tau = tune_conf_threshold(model, val_loader, DEVICE)
        ckpt['best_conf_tau'] = best_conf_tau
        ckpt = checkpoint_mark(ckpt, 'conf_tau')
        save_checkpoint(ckpt)

    if checkpoint_has(ckpt, 'iees_tau'):
        best_iees_thresholds = ckpt['best_iees_thresholds']
        print(f'  ✅ IEES-proxy tau: {best_iees_thresholds[0]:.2f}')
    else:
        best_iees_thresholds = tune_proxy_thresholds(model, rf, val_loader, DEVICE)
        ckpt['best_iees_thresholds'] = best_iees_thresholds
        ckpt = checkpoint_mark(ckpt, 'iees_tau')
        save_checkpoint(ckpt)

    # Step 5: Evaluate on test.
    print('\nStep 5: Evaluating on test...')
    results: Dict[str, Tuple] = {}

    eval_specs = [
        ('No-early-exit', 'eval_no_exit', lambda: run_no_early_exit(model, test_loader, DEVICE)),
        ('Conf. (Optimized)', 'eval_conf_opt', lambda: run_confidence_policy(model, test_loader, DEVICE, best_conf_tau)),
        ('IEES-proxy (Optimized)', 'eval_iees_proxy', lambda: run_proxy_policy(model, rf, test_loader, DEVICE, best_iees_thresholds)),
    ]

    for policy, key, fn in eval_specs:
        if checkpoint_has(ckpt, key):
            r = ckpt[key]
            print(f'  ✅ {policy} cached: {r[0]:.2f}%')
        else:
            print(f'  Running {policy}...')
            r = list(fn())
            ckpt[key] = r
            ckpt = checkpoint_mark(ckpt, key)
            save_checkpoint(ckpt)
            print(f'  ✅ {policy}: {r[0]:.2f}%')
        results[policy] = (r[0], r[1], r[2], r[3], 0.0)

    # Matched confidence must be matched to IEES distribution on VALIDATION only.
    val_dist_key = 'val_iees_proxy_dist'
    if checkpoint_has(ckpt, val_dist_key):
        val_iees_dist = ckpt[val_dist_key]
        print('  ✅ Validation IEES-proxy distribution cached.')
    else:
        print('  Computing IEES-proxy exit distribution on validation for matched-confidence baseline...')
        _, val_iees_dist, _, _ = run_proxy_policy(model, rf, val_loader, DEVICE, best_iees_thresholds)
        ckpt[val_dist_key] = val_iees_dist
        ckpt = checkpoint_mark(ckpt, val_dist_key)
        save_checkpoint(ckpt)

    policy = 'Conf. (IEES-matched)'
    key = 'eval_conf_matched'
    if checkpoint_has(ckpt, key):
        r = ckpt[key]
        print(f'  ✅ {policy} cached: {r[0]:.2f}%')
    else:
        print(f'  Running {policy}...')
        r = list(find_matched_confidence(model, val_loader, test_loader, DEVICE, val_iees_dist))
        ckpt[key] = r
        ckpt = checkpoint_mark(ckpt, key)
        save_checkpoint(ckpt)
        print(f'  ✅ {policy}: {r[0]:.2f}%')
    results[policy] = (r[0], r[1], r[2], r[3], 0.0)

    print_results(results, reported_acc)

    save_path = os.path.join(RESULTS_DIR, 'dvt_iees_pfam_synced_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'DVT_T2t_vit_12',
            'setting': 'Oracle PFAM/IEES offline supervision + attribution-free IEES proxy at inference',
            'iees_formula': 'IEES_i = w1*S_i + w2*F_i - w3*Q_i',
            'pfam_common_resolution': COMMON_RESOLUTION,
            'top_k_percent_for_F': TOP_K_PERCENT,
            'weights': IEES_WEIGHTS,
            'normalization': ckpt.get('normalization', {}),
            'reported_acc': reported_acc,
            'quick_acc': quick_acc,
            'r2_val': r2,
            'mae_val': mae,
            'results': {
                k: {'acc': v[0], 'dist': v[1], 'exit_acc': v[2], 'time': v[3]}
                for k, v in results.items()
            },
        }, f, indent=2)

    print(f'\n✅ Results → {save_path}')
    ckpt = checkpoint_mark(ckpt, 'complete')
    save_checkpoint(ckpt)
    print('✅ DVT experiment complete!')


if __name__ == '__main__':
    main()
