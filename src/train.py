from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, optim

from labels import get_class_label
from src.iees_utils import (
    DEFAULT_COMMON_RESOLUTION,
    calculate_consistency_index,
    compute_attribution_strength,
    generate_pfams,
    minmax_normalize,
)
from src.pfam_utils import ExitAttributionCache, generate_pfams_after_exit
from src.proxy import ProxyFeatureState, get_num_decision_exits, proxy_features_for_runtime_exit
from src.visuals import plot_combined_image, visualize_xai
from src.xai import getAttribution
from util import getCumulativeFlops, load_model, reorganize_attribution_maps, generate_thresholds


def load_and_train_model(model_name, num_c, num_exits, device, deployment, num_epochs, trainloader, testloader):
    model = load_model(model_name, num_c, num_exits).to(device)
    if not deployment:
        print("Training started!")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            train(model, trainloader, criterion, optimizer, scheduler, device, epoch, [1.0, 0.7, 0.5, 0.3])
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), f"../models/{model_name}.pth")
    else:
        model.load_state_dict(torch.load(f"../models/{model_name}.pth", map_location=device, weights_only=True))
    return model


def train(model, dataloader, criterion, optimizer, scheduler, device, epoch, loss_weights):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total_loss = 0.0
        for i, exit_output in enumerate(outputs):
            w = loss_weights[i] if i < len(loss_weights) else 1.0
            total_loss = total_loss + w * criterion(exit_output, labels)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        running_loss += float(total_loss.item())
        if batch_idx % 100 == 99:
            print(f"Epoch [{epoch}], Batch [{batch_idx + 1}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
    scheduler.step()


def _as_thresholds(thresholds, n):
    if thresholds is None:
        return [0.7] * n
    if isinstance(thresholds, (float, int)):
        return [float(thresholds)] * n
    thresholds = list(thresholds)
    if len(thresholds) < n:
        thresholds = thresholds + [thresholds[-1]] * (n - len(thresholds))
    return thresholds[:n]


def _confidence_from_logits(logits):
    probs = F.softmax(logits, dim=1)
    conf = float(probs.max(dim=1).values.item())
    pred = int(probs.argmax(dim=1).item())
    return conf, pred


def inference(model, proxy, testloader, device, default_thresholds, dataset_name, output_dir, weights=None, generate_visuals=False):
    """Run deployment inference with the trained IEES proxy.

    No PFAM/Grad-CAM is computed for exit decisions. Set ``generate_visuals=True``
    only for analysis/visualization after the exit has been selected.
    """
    model.eval()
    n_exits = get_num_decision_exits(model)
    thresholds = _as_thresholds(default_thresholds, n_exits)
    os.makedirs(f"{output_dir}/pfam", exist_ok=True)

    correct = total = 0
    exit_counts = [0] * n_exits

    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        for i in range(inputs.size(0)):
            input_image = inputs[i : i + 1]
            label = int(labels[i].item())
            prev_state: Optional[ProxyFeatureState] = None
            predicted_classes, iees_scores, confidence_scores = [], [], []
            cache = ExitAttributionCache()
            exit_idx = n_exits - 1
            pred = None

            for idx in range(n_exits):
                logits, class_idx, _ = model.forward_to_exit(input_image, idx,False)
                activations = getattr(model, "activations", None)
                feat, prev_state = proxy_features_for_runtime_exit(logits, activations, prev_state)
                pred_iees = float(proxy.predict(feat.reshape(1, -1))[0])
                conf, pred = _confidence_from_logits(logits)
                predicted_classes.append(pred)
                iees_scores.append(pred_iees)
                confidence_scores.append(conf)
                if generate_visuals:
                    cache.cache(idx, model, logits, class_idx)
                if pred_iees >= thresholds[idx] or idx == n_exits - 1:
                    exit_idx = idx
                    break

            correct += int(pred == label)
            total += 1
            exit_counts[exit_idx] += 1

            if generate_visuals:
                input_image_np = input_image.cpu().detach().squeeze().permute(1, 2, 0).numpy()
                pfams_list, cumulative_maps, _ = generate_pfams_after_exit(
                    model, input_image, exit_idx, cache.activations, cache.outputs, cache.class_ids
                )
                final_cumulative = cumulative_maps[-1] if isinstance(cumulative_maps, list) else cumulative_maps
                filename = f"{output_dir}/pfam/exit_{exit_idx}.png"
                plot_combined_image(
                    input_image_np,
                    pfams_list,
                    final_cumulative,
                    filename,
                    get_class_label(dataset_name, predicted_classes),
                    get_class_label(dataset_name, label),
                    iees_scores,
                    confidence_scores,
                )

    acc = correct / max(total, 1)
    print(f"[Inference] Accuracy={100*acc:.2f}% | Exit counts={exit_counts}")
    return {"accuracy": acc, "exit_counts": exit_counts}


def comparision_with_XAI_tools(model, dataset_name, testloader, device, default_thresholds, weights):
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        for i in range(inputs.size(0)):
            input_image = inputs[i : i + 1]
            label = labels[i].item()
            pfams_list, cumulative_map_fpam, cumulative_maps, predicted_classes, all_scores, all_attributions = getAttribution(
                model, input_image, default_thresholds
            )
            all_attributions = reorganize_attribution_maps(all_attributions)
            visualize_xai(
                input_image,
                pfams_list,
                all_attributions,
                cumulative_maps,
                cumulative_map_fpam,
                "jet",
                get_class_label(dataset_name, predicted_classes),
                get_class_label(dataset_name, label),
            )


def _extract_oracle_iees_records(model, dataloader, device, weights=(0.5, 0.3, 0.2), k=20, max_samples=None):
    """Offline oracle IEES records with final paper S/F/Q definitions."""
    model.eval()
    n_exits = get_num_decision_exits(model)
    records = []
    F_all, Q_all = [], []
    seen = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        for b in range(inputs.size(0)):
            if max_samples is not None and seen >= max_samples:
                break
            img, label = inputs[b : b + 1], int(labels[b].item())
            sample = []
            pfams_list, cumulative_maps, channel_weighted_maps = [], [], []
            for exit_idx in range(n_exits):
                model.zero_grad(set_to_none=True)
                logits, class_idx, _ = model.forward_to_exit(img, exit_idx, True)
                probs = F.softmax(logits, dim=1)
                pred = int(probs.argmax(dim=1).item())
                S = float(probs.max(dim=1).values.item())
                pfams_list, cumulative_map, channel_weighted_maps = generate_pfams(
                    model.activations,
                    model.gradients,
                    img,
                    pfams_list,
                    channel_weighted_maps,
                    common_resolution=DEFAULT_COMMON_RESOLUTION,
                )
                cumulative_maps.append(cumulative_map)
                Q_raw = float(calculate_consistency_index(cumulative_maps))
                F_raw = float(compute_attribution_strength(channel_weighted_maps[-1], k=k))
                F_all.append(F_raw); Q_all.append(Q_raw)
                sample.append({"logits": logits.detach().cpu(), "pred": pred, "label": label, "S": S, "F_raw": F_raw, "Q_raw": Q_raw})
                model.activations = None; model.gradients = None
            records.append(sample)
            seen += 1
        if max_samples is not None and seen >= max_samples:
            break

    F_norm = minmax_normalize(F_all)
    Q_norm = minmax_normalize(Q_all)
    idx = 0
    w1, w2, w3 = [float(x) for x in weights]
    for sample in records:
        for rec in sample:
            rec["F"] = float(F_norm[idx]); rec["Q"] = float(Q_norm[idx])
            rec["iees"] = w1 * rec["S"] + w2 * rec["F"] - w3 * rec["Q"]
            idx += 1
    return records


def distribution_drift(model, model_name, testloader, device, exit_criterion="confidence", threshold_range=None, weights=None, proxy=None):
    bbc_results = evaluate_model_thresholds(model, model_name, testloader, device, exit_criterion, threshold_range, weights or [], proxy=proxy)[0]
    return {
        'threshold': bbc_results["threshold"],
        'accuracy': bbc_results["avg_accuracy"],
        'avg_flops': bbc_results["avg_flops"],
        'exit_counts': bbc_results["exit_counts"],
        'avg_latencies': bbc_results["avg_latencies"],
        'exit_accuracies': bbc_results["exit_accuracies"],
    }


def evaluate_model_thresholds(model, model_name, testloader, device, exit_criterion="confidence", threshold=None, weights=None, proxy=None):
    """Evaluate confidence, proxy IEES, or offline oracle IEES thresholds.

    For paper-reported deployment results, pass ``exit_criterion='iees'`` with
    a trained ``proxy``. If no proxy is provided, oracle IEES is computed offline
    using PFAM/Grad-CAM for ablation/calibration analysis only.
    """
    weights = weights or (0.5, 0.3, 0.2)
    n_exits = get_num_decision_exits(model)
    threshold_range = generate_thresholds() if threshold is None else (threshold if isinstance(threshold, list) else [threshold])
    flops_lookup = getCumulativeFlops(model_name, baseline=False)
    all_results = []

    oracle_records = None
    if exit_criterion == "iees" and proxy is None:
        print("[INFO] No proxy provided; computing offline oracle IEES for analysis/ablation.")
        oracle_records = _extract_oracle_iees_records(model, testloader, device, weights)

    for threshold_val in threshold_range:
        thresholds = _as_thresholds(threshold_val, n_exits)
        correct_preds = total_samples = total_flops = 0
        exit_correct_preds = [0] * n_exits
        exit_counts = [0] * n_exits
        exit_latencies = [0.0] * n_exits

        if oracle_records is not None:
            for sample in oracle_records:
                start = time.time(); chosen = n_exits - 1
                for exit_idx, rec in enumerate(sample):
                    if rec["iees"] >= thresholds[exit_idx] or exit_idx == n_exits - 1:
                        chosen = exit_idx; pred = rec["pred"]; label = rec["label"]; break
                latency = (time.time() - start) * 1000
                correct = int(pred == label)
                correct_preds += correct; total_samples += 1
                exit_counts[chosen] += 1; exit_correct_preds[chosen] += correct
                exit_latencies[chosen] += latency
                if chosen < len(flops_lookup): total_flops += flops_lookup[chosen]
            
        else:
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    for b in range(inputs.size(0)):
                        img, label = inputs[b : b + 1], int(labels[b].item())
                        prev_state: Optional[ProxyFeatureState] = None
                        chosen = n_exits - 1; pred = None
                        start = time.time()
                        for exit_idx in range(n_exits):
                            logits, _, _ = model.forward_to_exit(img, exit_idx, False)
                            conf, pred = _confidence_from_logits(logits)
                            if exit_criterion == "confidence":
                                score = conf
                            elif exit_criterion == "iees" and proxy is not None:
                                feat, prev_state = proxy_features_for_runtime_exit(logits, getattr(model, "activations", None), prev_state)
                                score = float(proxy.predict(feat.reshape(1, -1))[0])
                            else:
                                raise ValueError("exit_criterion must be 'confidence' or 'iees'")
                            if score >= thresholds[exit_idx] or exit_idx == n_exits - 1:
                                chosen = exit_idx; break
                        latency = (time.time() - start) * 1000
                        correct = int(pred == label)
                        correct_preds += correct; total_samples += 1
                        exit_counts[chosen] += 1; exit_correct_preds[chosen] += correct
                        exit_latencies[chosen] += latency
                        if chosen < len(flops_lookup): total_flops += flops_lookup[chosen]

        avg_acc = correct_preds / max(total_samples, 1)
        avg_flops = total_flops / max(total_samples, 1)
        avg_latencies = [lat / count if count > 0 else 0.0 for lat, count in zip(exit_latencies, exit_counts)]
        exit_accuracies = [exit_correct_preds[i] / exit_counts[i] if exit_counts[i] > 0 else 0.0 for i in range(n_exits)]
        weighted_overall_accuracy = sum(exit_accuracies[i] * (exit_counts[i] / max(total_samples, 1)) for i in range(n_exits))
        exit_mean_accuracy = sum(exit_accuracies) / n_exits
        all_results.append({
            'threshold': threshold_val,
            'avg_accuracy': avg_acc,
            'avg_flops': avg_flops,
            'exit_counts': exit_counts,
            'avg_latencies': avg_latencies,
            'exit_accuracies': exit_accuracies,
            'weighted_overall_accuracy': weighted_overall_accuracy,
            'exit_mean_accuracy': exit_mean_accuracy,
        })
    return all_results
