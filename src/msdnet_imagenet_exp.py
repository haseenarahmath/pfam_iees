# src/msdnet_imagenet_exp_final.py
# MSDNet / ImageNet experiment aligned with updated IEES + PFAM paper formulation.
#
# Key alignment points:
#   - Oracle IEES is computed offline from PFAM/Grad-CAM attribution components.
#   - Runtime policy uses an attribution-free Random Forest proxy.
#   - IEES_i(x) = w1*S_i(x) + w2*F_i(x) - w3*Q_i(x).
#   - F_i uses raw pre-ReLU channel-weighted attribution strength over top-k% salient locations.
#   - Q_i is mean absolute change between consecutive cumulative PFAM maps.
#   - F_i and Q_i are min-max normalized using calibration statistics before IEES scoring.
#   - Confidence-matched baseline is matched on validation only to avoid test leakage.
#   - Validation/test policies use batch_size=1 for per-sample early-exit decisions.

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
from sklearn.metrics import r2_score

from src.msdnet_imagenet import MSDNetImageNet
from src.iees_utils import (
    DEFAULT_COMMON_RESOLUTION,
    generate_pfams,
    calculate_consistency_index,
    compute_attribution_strength,
)

# ── Config ────────────────────────────────────────────────────────────────────
IMAGENET_DIR = os.environ.get('IMAGENET_DIR', '../data/imagenet')
CHECKPOINT = os.environ.get('MSDNET_CHECKPOINT', '../models/msdnet_imagenet.pth')

RESULTS_DIR = '../results'
FEATURES_DIR = '../results/msdnet_imagenet_iees_pfam_final_features'
CHECKPOINT_FILE = '../results/msdnet_imagenet_iees_pfam_final_checkpoint.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE_CAL = 32
BATCH_SIZE_EVAL = 1  # per-sample early-exit decisions
CAL_SIZE = 25000
VAL_SIZE = 12000
TEST_SIZE = 12000
SEED = 42
NUM_EXITS = 5
NUM_RUNS = 1
IEES_WEIGHTS = (0.5, 0.3, 0.2)
TOP_K_PERCENT = 20
COMMON_RESOLUTION = DEFAULT_COMMON_RESOLUTION

print(f'Device: {DEVICE}')
print(f'NUM_EXITS: {NUM_EXITS}')


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(data: Dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data['last_updated'] = datetime.datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'  💾 Checkpoint saved → {CHECKPOINT_FILE}')


def load_checkpoint() -> Optional[Dict]:
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE, encoding='utf-8') as f:
        data = json.load(f)
    print('  ⚡ Checkpoint found!')
    print(f"     Completed steps: {data.get('completed_steps', [])}")
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


# ── Feature/component persistence ─────────────────────────────────────────────
def _prefix(split_name: str) -> str:
    return os.path.join(FEATURES_DIR, split_name)


def save_components(split_name: str,
                    X: np.ndarray,
                    S: np.ndarray,
                    F_raw: np.ndarray,
                    Q_raw: np.ndarray,
                    y_corr: np.ndarray,
                    exits: np.ndarray) -> None:
    os.makedirs(FEATURES_DIR, exist_ok=True)
    prefix = _prefix(split_name)
    np.save(f'{prefix}_X.npy', X)
    np.save(f'{prefix}_S.npy', S)
    np.save(f'{prefix}_F_raw.npy', F_raw)
    np.save(f'{prefix}_Q_raw.npy', Q_raw)
    np.save(f'{prefix}_y_corr.npy', y_corr)
    np.save(f'{prefix}_exits.npy', exits)
    with open(f'{prefix}_meta.json', 'w', encoding='utf-8') as f:
        json.dump({
            'split': split_name,
            'n_exit_samples': int(len(X)),
            'n_features': int(X.shape[1]),
            'n_exits': NUM_EXITS,
            'top_k_percent': TOP_K_PERCENT,
            'common_resolution': list(COMMON_RESOLUTION),
            'note': 'Raw oracle IEES components. F/Q are normalized later using calibration statistics.'
        }, f, indent=2)
    mb = (X.nbytes + S.nbytes + F_raw.nbytes + Q_raw.nbytes + y_corr.nbytes + exits.nbytes) / 1024 / 1024
    print(f'  ✅ Saved {split_name}: {len(X)} exit-samples ({mb:.1f} MB)')


def load_components(split_name: str):
    prefix = _prefix(split_name)
    required = [
        f'{prefix}_X.npy', f'{prefix}_S.npy', f'{prefix}_F_raw.npy',
        f'{prefix}_Q_raw.npy', f'{prefix}_y_corr.npy', f'{prefix}_exits.npy'
    ]
    if not all(os.path.exists(p) for p in required):
        return None
    X = np.load(f'{prefix}_X.npy')
    S = np.load(f'{prefix}_S.npy')
    F_raw = np.load(f'{prefix}_F_raw.npy')
    Q_raw = np.load(f'{prefix}_Q_raw.npy')
    y_corr = np.load(f'{prefix}_y_corr.npy')
    exits = np.load(f'{prefix}_exits.npy')
    print(f'  ✅ Loaded {split_name}: {len(X)} exit-samples')
    return X, S, F_raw, Q_raw, y_corr, exits


def components_exist(split_name: str) -> bool:
    return os.path.exists(f'{_prefix(split_name)}_X.npy')


# ── Data/model loading ────────────────────────────────────────────────────────
def build_label_remap(dataset: datasets.ImageFolder) -> Dict[int, int]:
    standard = list(ResNet18_Weights.IMAGENET1K_V1.meta['categories'])

    def norm(s: str) -> str:
        return s.lower().replace(' ', '_').replace('-', '_').replace("'", '_').replace(',', '')

    std_lookup = {norm(name): idx for idx, name in enumerate(standard)}
    remap: Dict[int, int] = {}
    not_found: List[str] = []

    for your_idx, cls in enumerate(dataset.classes):
        key = norm(cls)
        if key in std_lookup:
            remap[your_idx] = std_lookup[key]
        else:
            not_found.append(cls)
            remap[your_idx] = your_idx

    print(f'Label remap: {len(remap) - len(not_found)}/{len(remap)} classes matched')
    if not_found:
        print(f'  ⚠️ Unmatched classes (showing up to 10): {not_found[:10]}')
    return remap


def load_imagenet_splits(data_dir: str):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    print(f'Loading ImageNet from {data_dir}')
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f'Total images: {len(dataset)}')

    remap = build_label_remap(dataset)
    dataset.targets = [remap[t] for t in dataset.targets]
    dataset.samples = [(p, remap[l]) for p, l in dataset.samples]

    total_needed = CAL_SIZE + VAL_SIZE + TEST_SIZE
    torch.manual_seed(SEED)
    indices = torch.randperm(len(dataset))[:total_needed].tolist()

    cal_set = Subset(dataset, indices[:CAL_SIZE])
    val_set = Subset(dataset, indices[CAL_SIZE:CAL_SIZE + VAL_SIZE])
    test_set = Subset(dataset, indices[CAL_SIZE + VAL_SIZE:])

    print(f'  Calibration: {len(cal_set)}')
    print(f'  Validation:  {len(val_set)}')
    print(f'  Test:        {len(test_set)}')

    cal_loader = DataLoader(cal_set, batch_size=BATCH_SIZE_CAL, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=0)
    return cal_loader, val_loader, test_loader


def load_msdnet(checkpoint_path: str, device: torch.device):
    print('Building MSDNet...')
    model = MSDNetImageNet(num_classes=1000, num_exits=NUM_EXITS)

    print('Loading pretrained weights...')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if not missing and not unexpected:
        print('✅ Perfect weight match!')
    else:
        print(f'  Missing: {len(missing)}, Unexpected: {len(unexpected)}')

    acc = 100.0 - float(ckpt.get('best_err1', 0.0))
    print(f'✅ Loaded! Reported/best acc: {acc:.2f}%')
    model = model.to(device)
    model.eval()
    return model, acc


# ── IEES/proxy feature helpers ────────────────────────────────────────────────
def normalize_with_stats(values: np.ndarray, lo: float, hi: float, eps: float = 1e-8) -> np.ndarray:
    return ((values - lo) / (hi - lo + eps)).clip(0.0, 1.0)


def build_iees_targets(S: np.ndarray,
                       F_raw: np.ndarray,
                       Q_raw: np.ndarray,
                       norm_stats: Dict[str, float],
                       weights: Tuple[float, float, float]) -> np.ndarray:
    w1, w2, w3 = weights
    F_norm = normalize_with_stats(F_raw, norm_stats['F_min'], norm_stats['F_max'])
    Q_norm = normalize_with_stats(Q_raw, norm_stats['Q_min'], norm_stats['Q_max'])
    return (w1 * S + w2 * F_norm - w3 * Q_norm).astype(np.float32)


def sample_proxy_features_from_logits(logits: torch.Tensor,
                                      activations: Optional[torch.Tensor],
                                      prev_state: Optional[Dict]) -> Tuple[np.ndarray, Dict]:
    """Ten attribution-free proxy features, matching Appendix B."""
    probs = F.softmax(logits, dim=1)
    conf = float(probs.max(dim=1).values.item())
    entropy = float(-(probs * torch.log(probs + 1e-8)).sum().item())
    top2 = torch.topk(probs, 2, dim=1).values
    margin = float((top2[:, 0] - top2[:, 1]).item())
    pred = int(probs.argmax(dim=1).item())

    if activations is not None:
        act_mean = float(activations.abs().mean().item())
        act_max = float(activations.abs().max().item())
    else:
        act_mean = float(logits.abs().mean().item())
        act_max = float(logits.abs().max().item())

    if prev_state is None:
        conf_delta = 0.0
        class_stability = 0.0
        logit_cos = 0.0
        entropy_delta = 0.0
        act_delta = 0.0
    else:
        conf_delta = abs(conf - float(prev_state['conf']))
        class_stability = float(pred == int(prev_state['pred']))
        logit_cos = float(F.cosine_similarity(logits.detach(), prev_state['logits'], dim=1).item())
        entropy_delta = abs(entropy - float(prev_state['entropy']))
        act_delta = abs(act_mean - float(prev_state['act_mean']))

    feat = np.array([
        conf, margin, entropy, act_mean, act_max,
        conf_delta, class_stability, logit_cos, entropy_delta, act_delta
    ], dtype=np.float32)

    state = {
        'conf': conf,
        'pred': pred,
        'logits': logits.detach(),
        'entropy': entropy,
        'act_mean': act_mean,
    }
    return feat, state


def extract_oracle_components(model,
                              loader: DataLoader,
                              device: torch.device,
                              desc: str = 'Extracting'):
    """
    Offline oracle phase.

    Computes PFAM/Grad-CAM based F_i and Q_i, plus the 10 attribution-free
    features used to train the deployment proxy.
    """
    all_features: List[np.ndarray] = []
    all_S: List[float] = []
    all_F_raw: List[float] = []
    all_Q_raw: List[float] = []
    all_correct: List[int] = []
    all_exits: List[int] = []

    print(f'{desc} oracle IEES components...')
    print('  PFAM/Grad-CAM used offline only; runtime uses proxy features.')

    model.eval()
    for batch_idx, (imgs, labels) in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(loader)}')

        imgs = imgs.to(device)
        labels = labels.to(device)

        for b in range(imgs.size(0)):
            img = imgs[b:b + 1]
            label = labels[b]
            pfams_list: List[np.ndarray] = []
            cumulative_maps: List[np.ndarray] = []
            channel_weighted_maps: List[np.ndarray] = []
            prev_state: Optional[Dict] = None

            for exit_idx in range(NUM_EXITS):
                model.zero_grad(set_to_none=True)

                logits, class_idx, target = model.forward_to_exit(
                    img, exit_idx, gradient_required=True
                )

                # forward_to_exit is expected to set model.activations and model.gradients.
                activations = model.activations
                gradients = model.gradients

                if activations is None or gradients is None:
                    raise RuntimeError(
                        'model.forward_to_exit did not populate activations/gradients. '
                        'Check MSDNetImageNet.forward_to_exit hooks.'
                    )

                pfams_list, cumulative_map, channel_weighted_maps = generate_pfams(
                    activations,
                    gradients,
                    img,
                    pfams_list,
                    channel_weighted_maps,
                    common_resolution=COMMON_RESOLUTION,
                )
                cumulative_maps.append(cumulative_map)

                Q_raw = calculate_consistency_index(cumulative_maps)
                F_raw = compute_attribution_strength(channel_weighted_maps[-1], k=TOP_K_PERCENT)

                probs = F.softmax(logits, dim=1)
                S = float(probs.max(dim=1).values.item())
                pred = int(probs.argmax(dim=1).item())
                correct = int(pred == int(label.item()))

                feat, prev_state = sample_proxy_features_from_logits(logits, activations, prev_state)

                all_features.append(feat)
                all_S.append(S)
                all_F_raw.append(float(F_raw))
                all_Q_raw.append(float(Q_raw))
                all_correct.append(correct)
                all_exits.append(exit_idx)

                model.activations = None
                model.gradients = None

    X = np.asarray(all_features, dtype=np.float32)
    S = np.asarray(all_S, dtype=np.float32)
    F_raw = np.asarray(all_F_raw, dtype=np.float32)
    Q_raw = np.asarray(all_Q_raw, dtype=np.float32)
    y_corr = np.asarray(all_correct, dtype=np.float32)
    exits = np.asarray(all_exits, dtype=np.int32)

    n_img = len(X) // NUM_EXITS
    print(f'  Extracted: {len(X)} exit-samples ({n_img} images × {NUM_EXITS} exits)')
    print(f'  F_raw range: [{F_raw.min():.4f}, {F_raw.max():.4f}]')
    print(f'  Q_raw range: [{Q_raw.min():.4f}, {Q_raw.max():.4f}]')
    return X, S, F_raw, Q_raw, y_corr, exits


# ── Runtime policies: attribution-free ───────────────────────────────────────
def run_no_early_exit(model, loader: DataLoader, device: torch.device):
    correct = 0
    total = 0
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            t0 = time.perf_counter()
            all_logits = model(imgs)
            logits = all_logits[-1]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))
            total_time += (t1 - t0) * 1000.0

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
            imgs = imgs.to(device)
            labels = labels.to(device)
            assert imgs.size(0) == 1, 'Use batch_size=1 for per-sample early-exit evaluation.'

            t0 = time.perf_counter()
            all_logits = model(imgs)
            for exit_idx, logits in enumerate(all_logits):
                probs = F.softmax(logits, dim=1)
                conf = float(probs.max(dim=1).values.item())

                if conf >= threshold or exit_idx == NUM_EXITS - 1:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    pred = int(logits.argmax(dim=1).item())
                    is_correct = int(pred == int(labels.item()))
                    correct += is_correct
                    total += 1
                    exit_counts[exit_idx] += 1
                    exit_correct[exit_idx] += is_correct
                    total_time += (t1 - t0) * 1000.0
                    break

    acc = 100.0 * correct / total
    dist = [100.0 * c / total for c in exit_counts]
    exit_acc = [100.0 * exit_correct[i] / exit_counts[i] if exit_counts[i] else 0.0 for i in range(NUM_EXITS)]
    return acc, dist, exit_acc, total_time / total


def run_proxy_policy(model, rf: RandomForestRegressor, loader: DataLoader, device: torch.device, thresholds: List[float]):
    """Deployment IEES policy: no PFAM/Grad-CAM/backward pass at inference."""
    correct = 0
    total = 0
    exit_counts = [0] * NUM_EXITS
    exit_correct = [0] * NUM_EXITS
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            assert imgs.size(0) == 1, 'Use batch_size=1 for per-sample early-exit evaluation.'

            t0 = time.perf_counter()
            prev_state: Optional[Dict] = None

            # Sequentially call forward_to_exit to expose activations needed by the 10 proxy features.
            # No gradients/attributions are computed here.
            for exit_idx in range(NUM_EXITS):
                logits, _, _ = model.forward_to_exit(imgs, exit_idx, gradient_required=False)
                activations = model.activations
                feat, prev_state = sample_proxy_features_from_logits(logits, activations, prev_state)
                pred_score = float(rf.predict(feat.reshape(1, -1))[0])

                if pred_score >= thresholds[exit_idx] or exit_idx == NUM_EXITS - 1:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    pred = int(logits.argmax(dim=1).item())
                    is_correct = int(pred == int(labels.item()))
                    correct += is_correct
                    total += 1
                    exit_counts[exit_idx] += 1
                    exit_correct[exit_idx] += is_correct
                    total_time += (t1 - t0) * 1000.0
                    break

                model.activations = None
                model.gradients = None

    acc = 100.0 * correct / total
    dist = [100.0 * c / total for c in exit_counts]
    exit_acc = [100.0 * exit_correct[i] / exit_counts[i] if exit_counts[i] else 0.0 for i in range(NUM_EXITS)]
    return acc, dist, exit_acc, total_time / total


# ── Threshold tuning ──────────────────────────────────────────────────────────
def tune_conf_threshold(model, val_loader: DataLoader, device: torch.device) -> float:
    print('  Tuning confidence threshold on validation...')
    best_tau = 0.90
    best_acc = -1.0
    for tau in np.arange(0.50, 0.99, 0.05):
        acc, _, _, _ = run_confidence_policy(model, val_loader, device, float(tau))
        if acc > best_acc:
            best_acc = acc
            best_tau = float(tau)
    print(f'  Best conf tau: {best_tau:.2f} → val acc: {best_acc:.2f}%')
    return best_tau


def tune_iees_thresholds(model, rf: RandomForestRegressor, val_loader: DataLoader, device: torch.device) -> List[float]:
    print('  Tuning IEES proxy thresholds on validation...')
    best_thresholds = [0.7] * NUM_EXITS
    best_acc = -1.0
    for tau in np.arange(0.50, 0.95, 0.05):
        thresholds = [float(tau)] * NUM_EXITS
        acc, _, _, _ = run_proxy_policy(model, rf, val_loader, device, thresholds)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds
    print(f'  Best IEES tau: {best_thresholds[0]:.2f} → val acc: {best_acc:.2f}%')
    return best_thresholds


def find_matched_confidence(model,
                            val_loader: DataLoader,
                            test_loader: DataLoader,
                            device: torch.device,
                            val_iees_dist: List[float]):
    """Match confidence to IEES exit distribution using validation only, then evaluate once on test."""
    print('  Finding IEES-matched confidence threshold on validation...')
    best_tau = 0.5
    best_diff = float('inf')

    for tau in np.arange(0.10, 0.99, 0.02):
        _, dist, _, _ = run_confidence_policy(model, val_loader, device, float(tau))
        diff = sum(abs(dist[i] - val_iees_dist[i]) for i in range(NUM_EXITS))
        if diff < best_diff:
            best_diff = diff
            best_tau = float(tau)

    print(f'  Best matched conf tau: {best_tau:.2f} (val diff={best_diff:.2f}%)')
    return run_confidence_policy(model, test_loader, device, best_tau)


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_results(results: Dict[str, Tuple]) -> None:
    print('\n' + '=' * 116)
    print('MSDNet / ImageNet — IEES/PFAM-Synced Exit Policy Comparison')
    print('Oracle PFAM/IEES computed offline; proxy IEES used at inference.')
    print('=' * 116)
    print(f"{'Exit Policy':<26}", end='')
    for i in range(NUM_EXITS):
        print(f'  E{i+1}% ', end='')
    print('  ', end='')
    for i in range(NUM_EXITS):
        print(f' E{i+1}Acc ', end='')
    print(f"  {'Overall':>10}  {'Time':>10}")
    print('-' * 116)

    for name, data in results.items():
        acc, dist, exit_acc, t_mean, t_std, acc_std = data
        print(f'{name:<26}', end='')
        for d in dist:
            print(f' {d:6.1f}', end='') if d > 0 else print('      –', end='')
        print('  ', end='')
        for a in exit_acc:
            print(f' {a:6.1f}', end='') if a > 0 else print('      –', end='')
        print(f'  {acc:7.2f}±{acc_std:.2f}  {t_mean:7.2f}±{t_std:.2f}ms')
    print('=' * 116)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print('\n' + '=' * 72)
    print('MSDNet ImageNet Experiment — Final IEES/PFAM-Synced Version')
    print('=' * 72 + '\n')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    ckpt = load_checkpoint()
    if ckpt is None:
        print('  Starting fresh!')
        ckpt = {'completed_steps': []}

    cal_loader, val_loader, test_loader = load_imagenet_splits(IMAGENET_DIR)
    model, reported_acc = load_msdnet(CHECKPOINT, DEVICE)

    if not checkpoint_has(ckpt, 'acc_check'):
        print('\nQuick accuracy check...')
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(val_loader):
                if i >= 500:
                    break
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                outs = model(imgs)
                preds = outs[-1].argmax(dim=1)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))
        quick_acc = 100.0 * correct / max(total, 1)
        print(f'Quick acc (E{NUM_EXITS}): {quick_acc:.2f}%')
        if quick_acc < 50:
            print('⚠️  Accuracy too low. Check label mapping/checkpoint.')
            return
        ckpt['quick_acc'] = quick_acc
        ckpt = checkpoint_mark(ckpt, 'acc_check')
        save_checkpoint(ckpt)
    else:
        quick_acc = ckpt.get('quick_acc', 0.0)
        print(f'✅ Acc check cached: {quick_acc:.2f}%')

    # Step 1: Offline oracle components.
    print('\nStep 1: Offline oracle IEES components...')
    if components_exist('cal'):
        X_cal, S_cal, F_cal_raw, Q_cal_raw, y_corr_cal, exits_cal = load_components('cal')
    else:
        X_cal, S_cal, F_cal_raw, Q_cal_raw, y_corr_cal, exits_cal = extract_oracle_components(
            model, cal_loader, DEVICE, 'Calibration'
        )
        save_components('cal', X_cal, S_cal, F_cal_raw, Q_cal_raw, y_corr_cal, exits_cal)
        ckpt = checkpoint_mark(ckpt, 'cal_components')
        save_checkpoint(ckpt)

    if components_exist('val'):
        X_val, S_val, F_val_raw, Q_val_raw, y_corr_val, exits_val = load_components('val')
    else:
        X_val, S_val, F_val_raw, Q_val_raw, y_corr_val, exits_val = extract_oracle_components(
            model, val_loader, DEVICE, 'Validation'
        )
        save_components('val', X_val, S_val, F_val_raw, Q_val_raw, y_corr_val, exits_val)
        ckpt = checkpoint_mark(ckpt, 'val_components')
        save_checkpoint(ckpt)

    # Step 2: Normalization and IEES targets.
    norm_stats = {
        'F_min': float(F_cal_raw.min()),
        'F_max': float(F_cal_raw.max()),
        'Q_min': float(Q_cal_raw.min()),
        'Q_max': float(Q_cal_raw.max()),
    }
    y_iees_cal = build_iees_targets(S_cal, F_cal_raw, Q_cal_raw, norm_stats, IEES_WEIGHTS)
    y_iees_val = build_iees_targets(S_val, F_val_raw, Q_val_raw, norm_stats, IEES_WEIGHTS)
    ckpt['norm_stats'] = norm_stats

    # Step 3: Train RF proxy.
    print('\nStep 2: Training attribution-free RF proxy...')
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_cal, y_iees_cal)
    y_pred_train = rf.predict(X_cal)
    y_pred_val = rf.predict(X_val)
    r2_train = float(r2_score(y_iees_cal, y_pred_train))
    r2_val = float(r2_score(y_iees_val, y_pred_val))
    print(f'  R² train: {r2_train:.3f}')
    print(f'  R² val:   {r2_val:.3f}')
    ckpt['r2_train'] = r2_train
    ckpt['r2_val'] = r2_val
    ckpt = checkpoint_mark(ckpt, 'rf_trained')
    save_checkpoint(ckpt)

    # Step 4: Tune thresholds on validation.
    print('\nStep 3: Tuning thresholds...')
    if checkpoint_has(ckpt, 'conf_tau'):
        best_conf_tau = ckpt['best_conf_tau']
        print(f'  ✅ Conf tau cached: {best_conf_tau:.2f}')
    else:
        best_conf_tau = tune_conf_threshold(model, val_loader, DEVICE)
        ckpt['best_conf_tau'] = best_conf_tau
        ckpt = checkpoint_mark(ckpt, 'conf_tau')
        save_checkpoint(ckpt)

    if checkpoint_has(ckpt, 'iees_tau'):
        best_iees_thresholds = ckpt['best_iees_thresholds']
        print(f'  ✅ IEES tau cached: {best_iees_thresholds[0]:.2f}')
    else:
        best_iees_thresholds = tune_iees_thresholds(model, rf, val_loader, DEVICE)
        ckpt['best_iees_thresholds'] = best_iees_thresholds
        ckpt = checkpoint_mark(ckpt, 'iees_tau')
        save_checkpoint(ckpt)

    # Step 5: Evaluate.
    print(f'\nStep 4: Evaluating ({NUM_RUNS} run(s))...')
    all_runs: List[Dict[str, List]] = []

    for run in range(NUM_RUNS):
        run_key = f'run_{run}_complete'
        run_data_key = f'run_{run}_data'
        if checkpoint_has(ckpt, run_key):
            print(f'  ✅ Run {run + 1} cached!')
            all_runs.append(ckpt[run_data_key])
            continue

        run_result: Dict[str, List] = {}

        for policy, fn in [
            ('No-early-exit', lambda: run_no_early_exit(model, test_loader, DEVICE)),
            ('Conf. (Optimized)', lambda: run_confidence_policy(model, test_loader, DEVICE, best_conf_tau)),
            ('IEES (Optimized)', lambda: run_proxy_policy(model, rf, test_loader, DEVICE, best_iees_thresholds)),
        ]:
            p_key = f'run{run}_{policy}'
            if checkpoint_has(ckpt, p_key):
                r = ckpt[p_key]
                print(f'    ✅ {policy} cached!')
            else:
                print(f'    Running {policy}...')
                r_tuple = fn()
                r = [r_tuple[0], r_tuple[1], r_tuple[2], r_tuple[3]]
                ckpt[p_key] = r
                ckpt = checkpoint_mark(ckpt, p_key)
                save_checkpoint(ckpt)
                print(f'    ✅ {policy}: {r[0]:.2f}%')
            run_result[policy] = r

        # Compute IEES validation distribution for matched-confidence baseline.
        val_iees_key = f'run{run}_val_iees_dist'
        if val_iees_key in ckpt:
            val_iees_dist = ckpt[val_iees_key]
        else:
            print('    Computing IEES validation distribution for matched confidence...')
            _, val_iees_dist, _, _ = run_proxy_policy(model, rf, val_loader, DEVICE, best_iees_thresholds)
            ckpt[val_iees_key] = val_iees_dist
            save_checkpoint(ckpt)

        policy = 'Conf. (IEES-matched)'
        p_key = f'run{run}_{policy}'
        if checkpoint_has(ckpt, p_key):
            r = ckpt[p_key]
            print(f'    ✅ {policy} cached!')
        else:
            print(f'    Running {policy}...')
            r_tuple = find_matched_confidence(model, val_loader, test_loader, DEVICE, val_iees_dist)
            r = [r_tuple[0], r_tuple[1], r_tuple[2], r_tuple[3]]
            ckpt[p_key] = r
            ckpt = checkpoint_mark(ckpt, p_key)
            save_checkpoint(ckpt)
            print(f'    ✅ {policy}: {r[0]:.2f}%')
        run_result[policy] = r

        all_runs.append(run_result)
        ckpt[run_data_key] = run_result
        ckpt = checkpoint_mark(ckpt, run_key)
        save_checkpoint(ckpt)

    # Aggregate.
    policy_names = ['No-early-exit', 'Conf. (Optimized)', 'IEES (Optimized)', 'Conf. (IEES-matched)']
    final_results: Dict[str, Tuple] = {}
    for name in policy_names:
        accs = [r[name][0] for r in all_runs]
        dists = [r[name][1] for r in all_runs]
        exit_accs = [r[name][2] for r in all_runs]
        times = [r[name][3] for r in all_runs]

        acc_mean = float(np.mean(accs))
        acc_std = float(np.std(accs))
        t_mean = float(np.mean(times))
        t_std = float(np.std(times))
        dist_mean = [float(np.mean([d[i] for d in dists])) for i in range(NUM_EXITS)]
        exit_acc_mean = [float(np.mean([ea[i] for ea in exit_accs])) for i in range(NUM_EXITS)]
        final_results[name] = (acc_mean, dist_mean, exit_acc_mean, t_mean, t_std, acc_std)

    print_results(final_results)

    save_path = os.path.join(RESULTS_DIR, 'msdnet_imagenet_iees_pfam_final_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'MSDNetImageNet',
            'reported_best_acc': reported_acc,
            'quick_acc_check': quick_acc,
            'iees_weights': list(IEES_WEIGHTS),
            'top_k_percent': TOP_K_PERCENT,
            'common_resolution': list(COMMON_RESOLUTION),
            'norm_stats': norm_stats,
            'proxy_r2_train': r2_train,
            'proxy_r2_val': r2_val,
            'best_conf_tau': best_conf_tau,
            'best_iees_thresholds': best_iees_thresholds,
            'results': {
                k: {
                    'overall_acc': v[0],
                    'exit_dist': v[1],
                    'exit_acc': v[2],
                    'time_mean': v[3],
                    'time_std': v[4],
                    'acc_std': v[5],
                }
                for k, v in final_results.items()
            },
            'notes': [
                'Oracle PFAM/IEES components computed offline only.',
                'Runtime IEES uses attribution-free RF proxy features.',
                'Confidence-matched baseline matched to IEES validation distribution only.',
                'F and Q normalized using calibration-set statistics before IEES scoring.'
            ]
        }, f, indent=2)

    print(f'\n✅ Final results → {save_path}')
    ckpt = checkpoint_mark(ckpt, 'complete')
    save_checkpoint(ckpt)
    print('✅ Experiment complete!')


if __name__ == '__main__':
    main()
