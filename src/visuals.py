import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.ndimage import gaussian_filter1d

from src.util import get_optimal_global_threshold
from util import normalize_flops, getCumulativeFlops, norm_flops, get_bck_acc, get_optimal_threshold


def save_heatmap(heatmap, filename):
    heatmap_normalized = normalize_for_display(heatmap)
    plt.figure()
    plt.imshow(heatmap_normalized, cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


def normalize_for_display(data):
    """
    Normalize the data to the range [0, 1] for display.
    """
    data_min, data_max = np.min(data), np.max(data)
    if data_max - data_min > 1e-10:  # Avoid division by zero
        normalized_data = (data - data_min) / (data_max - data_min)
    else:
        normalized_data = np.zeros_like(data)  # If data range is very small, set to zero
    return normalized_data


# Function to plot combined images with overlayed heatmaps for each exit
def plot_combined_image(input_image, pfam_maps, cumulative_map, output_filename, class_labels, gt, iees_scores=None,Confidence_scores=None):
    """
    Plots the input image, PFAM heatmaps for each exit, and the cumulative map with interpretability scores.

    Args:
    - input_image: Original input image.
    - pfam_maps: List of PFAM heatmaps for each exit.
    - cumulative_map: The cumulative attribution map at the final chosen exit.
    - output_filename: Path to save the output visualization.
    - class_labels: List of predicted class labels at each exit.
    - gt: Ground truth label.
    - iees_scores: List of IEES scores for each exit (optional).
    """

    # Normalize input image for display
    input_image = normalize_for_display(input_image)

    # Set up the figure with enough subplots for the input image, each PFAM map, and the cumulative map
    fig, axes = plt.subplots(1, len(pfam_maps) + 2, figsize=(15, 5))  # +2 for input and cumulative map

    # Plot the original input image with ground truth label
    axes[0].imshow(input_image)
    axes[0].set_title("Input Image",fontsize=14)
    axes[0].axis("off")
    axes[0].text(0.5, -0.1, f"GT: {gt}", ha="center", va="top", transform=axes[0].transAxes, fontsize=14)

    # Plot each PFAM map with exit labels, class labels, and IEES scores below
    for i, (pfam, pred_class) in enumerate(zip(pfam_maps, class_labels)):
        pfam_normalized = normalize_for_display(pfam)
        axes[i + 1].imshow(input_image, alpha=0.6)  # Display input as background
        axes[i + 1].imshow(pfam_normalized, cmap="jet", alpha=0.5)  # Overlay heatmap
        axes[i + 1].set_title(f"Exit {i + 1}",fontsize=14)
        axes[i + 1].axis("off")

        # Add IEEScore and predicted class label below each PFAM map
        # Construct the annotation text for each exit
        label_text = f"Pred: {pred_class}"
        if Confidence_scores:
            label_text += f"\nConfidence: {Confidence_scores[i]:.2f}"
        if iees_scores:
            label_text += f"\nIEES: {(iees_scores[i]):.2f}"
        # Display the annotation text below each PFAM map
        axes[i + 1].text(0.5, -0.05, label_text, ha="center", va="top", transform=axes[i + 1].transAxes, fontsize=14)


    #print(cumulative_map)
    # Plot the cumulative map at the final exit
    cumulative_map_normalized = normalize_for_display(cumulative_map)
    #print("cumulative_map_normalized", cumulative_map_normalized)
    axes[-1].imshow(input_image, alpha=0.6)  # Background image
    axes[-1].imshow(cumulative_map_normalized, cmap="jet", alpha=0.5)  # Overlay cumulative heatmap
    axes[-1].set_title("Cumulative Map",fontsize=14)
    axes[-1].axis("off")

    # Save and display the plot
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


def overlay_heatmap_on_image(image, heatmap, alpha=0.5, cmap='jet'):
    """
    Overlay a heatmap onto an input image with a consistent size and normalization.
    """
    # Ensure the heatmap is squeezed to 2D before applying the color map
    heatmap_normalized = heatmap.squeeze()  # Remove any excess dimensions
    if heatmap_normalized.ndim > 2:
        heatmap_normalized = heatmap_normalized[0]  # In case there's still an extra dimension

    # Apply color map to the heatmap for visualization, resulting in an RGB heatmap
    color_map = plt.get_cmap(cmap)
    heatmap_colored = color_map(heatmap_normalized)[..., :3]  # Convert to RGB format

    # Resize heatmap_colored to exactly match the input image dimensions
    heatmap_colored_resized = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))

    # Overlay the heatmap on the image
    overlayed_image = alpha * heatmap_colored_resized + (1 - alpha) * image
    return np.clip(overlayed_image, 0, 1)



def improved_visualization(input_image, exit_heatmaps, cumulative_map, cmap,class_labels,gt,IeeScore=None):
    """
    Generate an improved visualization with overlayed heatmaps on the input image.
    """
    # Convert input image to numpy and normalize to [0, 1] for display
    input_image_np = input_image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    input_image_np = normalize_for_display(input_image_np)  # Ensure consistent normalization for input

    fig, axes = plt.subplots(1, len(exit_heatmaps) + 2, figsize=(15, 5))

    # Display the input image
    axes[0].imshow(input_image_np)
    axes[0].set_title(f"Input Image\n{gt}")
    axes[0].axis("off")

    # Display each exit heatmap overlayed on the input image
    for i, (heatmap,pred_class) in enumerate(zip(exit_heatmaps, class_labels)):
        overlayed = overlay_heatmap_on_image(input_image_np, heatmap, alpha=0.5, cmap=cmap)
        axes[i + 1].imshow(overlayed)
        axes[i + 1].set_title(f"Exit {i + 1}\n{pred_class}")
        axes[i + 1].axis("off")

    # Display cumulative map overlayed on the input image
    overlayed_cumulative = overlay_heatmap_on_image(input_image_np, cumulative_map, alpha=0.5, cmap=cmap)
    axes[-1].imshow(overlayed_cumulative)
    axes[-1].set_title("Cumulative Map")
    axes[-1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_attributions(input_image, attribution_maps, method_names):
    """
    Visualize multiple attributions on input image.

    Args:
        input_image (torch.Tensor): Original image tensor.
        attribution_maps (dict): Dictionary with method names as keys and attributions as values.
        method_names (list): List of method names to be displayed.
    """
    fig, axes = plt.subplots(1, len(attribution_maps) + 1, figsize=(12, 5))

    # Normalize and show the input image
    input_image_np = input_image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())
    axes[0].imshow(input_image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot each attribution method as overlay
    for i, method in enumerate(method_names):
        heatmap = attribution_maps[method].squeeze().cpu().numpy()

        # Ensure overlay shape compatibility
        overlay = overlay_heatmap_on_image(input_image_np, heatmap)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"{method} Attribution")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()

def set_axes_title(n_cols,axes,ground_truth,label):
    for i in range(n_cols):
        if i == 0:
            axes[-1, i].text(0.5, -0.3, f'GT: {ground_truth}', fontsize=14, ha='center',
                             va='center',
                             transform=axes[-1, i].transAxes)
        else:
            axes[-1, i].text(0.5, -0.3,
                             f'Pred: {label[i-1]}',
                             fontsize=14, ha='center',
                             va='center', transform=axes[-1, i].transAxes)

    # Add bottom labels for exit points
    for i in range(n_cols):
        axes[-1, i].set_xlabel(f'Exit {i}', fontsize=12)





def visualize_xai(input_image, pfam_maps, xai_maps, cumulative_maps, pfam_cumulative_map, cmap, pred_labels, gt,
                  highlight_proposed=True):
    # Convert input image for display

    input_image_np = input_image.cpu().detach().squeeze().permute(1, 2, 0).numpy()
    input_image_np = np.clip(input_image_np, 0, 1)  # Clip to valid range for display

    n_exit_points = len(xai_maps["IG"])  # Number of exits we have data for
    n_attribution_types = 5  # IG, SmoothGrad, Grad-CAM, Occlusion, PFAM
    attribution_methods = [
        ("IG", xai_maps.get("IG")),
        ("SmoothGrad", xai_maps.get("SmoothGrad")),
        ("GradCam", xai_maps.get("GradCam")),
        ("Occlusion", xai_maps.get("Occlusion")),
        ("PFAM", pfam_maps)
    ]

    # Create subplots with additional rows for cumulative PFAM maps
    fig, axes = plt.subplots(nrows=n_attribution_types + 1, ncols=n_exit_points + 1, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust layout spacing

    for j, (attr_name, attr_data) in enumerate(attribution_methods):
        # Plot the input image in the first column of each row
        ax = axes[j, 0]
        ax.imshow(input_image_np)
        ax.axis('off')

        if j == 0:
            ax.set_title('Input Image', fontsize=10)

        for i in range(n_exit_points):
            ax = axes[j, i + 1]
            attr_map = attr_data[i].squeeze()  # Ensure attr_map is 2D
            attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-5)  # Normalize to [0, 1]
            overlay_attr = overlay_heatmap_on_image(input_image_np, attr_map, alpha=0.5, cmap=cmap)

            # Highlight PFAM if enabled
            if highlight_proposed and attr_name == "PFAM":
                ax.imshow(overlay_attr, interpolation='none')
                for spine in ax.spines.values():
                    spine.set_edgecolor('blue')  # Proposed color
                    spine.set_linewidth(2)
            else:
                ax.imshow(overlay_attr)

            # Explicitly highlight the last two rows with a red border
            if j >= n_attribution_types - 2:  # Check if it's one of the last two rows
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')  # Highlight color for last two rows
                    spine.set_linewidth(3)  # Increase line width for better visibility

            ax.axis('off')
            if j == 0:
                ax.set_title(f'Exit {i + 1}', fontsize=10)

        # Set the label for the rows (XAI methods) in the first column
        axes[j, 0].text(-0.1, 0.5, attr_name, fontsize=10, ha='center', va='center', rotation=90,
                        transform=axes[j, 0].transAxes)

    # Additional row for cumulative PFAM map
    ax = axes[-1, 0]
    ax.imshow(input_image_np)
    ax.axis('off')
    ax.text(-0.1, 0.5, "Cumulative", fontsize=10, ha='center', va='center', rotation=90,
            transform=axes[-1, 0].transAxes)
    ax.text(0.5, -0.3, f'GT: {gt}\n', ha='center', fontsize=10, transform=ax.transAxes)

    for i in range(n_exit_points):
        ax = axes[-1, i + 1]
        cumulative_map_normalized = (pfam_cumulative_map[i] - pfam_cumulative_map[i].min()) / (
                pfam_cumulative_map[i].max() - pfam_cumulative_map[i].min() + 1e-5)
        overlay_cumulative = overlay_heatmap_on_image(input_image_np, cumulative_map_normalized, alpha=0.5, cmap=cmap)

        # Apply red border for cumulative map row
        for spine in ax.spines.values():
            spine.set_edgecolor('red')  # Highlight color for the last row
            spine.set_linewidth(3)  # Increase line width for visibility

        ax.imshow(overlay_cumulative)
        ax.axis('off')
        ax.text(0.5, -0.3, f"Pred: {pred_labels[i]}\n", ha='center', fontsize=10, transform=ax.transAxes)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.show()




def plot_bbc(model_name, iees_csv, conf_csv):
    # Load and sort CSVs
    iees_df = pd.read_csv(iees_csv).sort_values('avg_flops')
    conf_df = pd.read_csv(conf_csv).sort_values('avg_flops')

    # Normalize FLOPs and Accuracy for plotting (unit-based)
    iees_x, iees_y, unit = normalize_flops(model_name, iees_df)
    conf_x, conf_y, _ = normalize_flops(model_name, conf_df)

    # ✅ Apply Gaussian smoothing for visualization
    iees_y_smooth = gaussian_filter1d(iees_y, sigma=1.2)
    conf_y_smooth = gaussian_filter1d(conf_y, sigma=1.2)

    # Backbone FLOPs and Accuracy
    backbone_flops_raw = getCumulativeFlops(model_name, baseline=True)
    backbone_flops, unit = norm_flops(model_name, backbone_flops_raw)
    backbone_acc = get_bck_acc(model_name)

    # Compute AUC (max-normalized for comparison)
    def compute_normalized_auc(df):
        norm_flops = df['avg_flops'] / df['avg_flops'].max()
        norm_acc = df['avg_accuracy']
        return auc(norm_flops, norm_acc)

    iees_auc = compute_normalized_auc(iees_df)
    conf_auc = compute_normalized_auc(conf_df)

    # Get optimal thresholds
    opt_thresh = get_optimal_global_threshold(model_name)
    conf_opt_thresh = get_optimal_global_threshold(model_name + "_conf")

    # Get optimal rows
    opt_row = iees_df[iees_df['threshold'].round(4) == round(opt_thresh, 4)].iloc[0]
    conf_opt_row = conf_df[conf_df['threshold'].round(4) == round(conf_opt_thresh, 4)].iloc[0]

    opt_flops = opt_row['avg_flops'] / (1e9 if unit == 'G' else 1e6)
    opt_acc = opt_row['avg_accuracy'] * 100
    conf_opt_flops = conf_opt_row['avg_flops'] / (1e9 if unit == 'G' else 1e6)
    conf_opt_acc = conf_opt_row['avg_accuracy'] * 100

    # Plotting
    plt.figure(figsize=(4.8, 3.8))
    plt.plot(iees_x, iees_y_smooth, label='IEES (Ours)', color='royalblue', linewidth=2)
    plt.plot(conf_x, conf_y_smooth, label='Confidence-only', color='darkorange', linewidth=2, linestyle='--')

    # Highlight optimal τ* points
    plt.scatter(opt_flops, opt_acc, color='red', zorder=5, label='Optimal τ*')
    plt.scatter(conf_opt_flops, conf_opt_acc, color='red', zorder=5)

    # No-early-exit reference
    plt.axhline(y=backbone_acc, color='steelblue', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(x=backbone_flops, color='steelblue', linestyle='--', linewidth=1, alpha=0.5)
    plt.plot(backbone_flops, backbone_acc, marker='o', color='black', markersize=6, label="No-early-exit")

    # Labels and layout
    plt.xlabel(f"Average FLOPs ({unit}FLOPs)", fontsize=11)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='lower right', fontsize=9, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    # ✅ Optional AUC annotation


    # plt.text(0.03, 0.97,  # <-- Top-left (x=3%, y=97%)
    #          f"IEES AUC = {iees_auc:.3f}\nConf. AUC = {conf_auc:.3f}",
    #          transform=plt.gca().transAxes,
    #          fontsize=10, color='black',
    #          ha='left', va='top')
    plt.text(0.15, 0.03, f"IEES AUC = {iees_auc:.3f}\nConf. AUC = {conf_auc:.3f}",
             transform=plt.gca().transAxes,
             fontsize=10, color='black', ha='left', va='bottom')

    plt.tight_layout()

    # Save and show
    plt.savefig(f"../results/plots/bbc_plot_{model_name.lower()}.png", dpi=400)
    plt.show()

