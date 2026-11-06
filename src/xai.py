from captum.attr import Occlusion
from captum.attr import DeepLiftShap
from src.iees_utils import generate_pfams, compute_iees_score, calculate_consistency_index
from visuals import visualize_attributions
import numpy as np
import torch
from captum.attr import LayerGradCam
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn



def get_combined_model(name, model, exit_index):
    combined_model = None
    if name == "msdnet":
        if exit_index < len(model.exits):
            combined_model = torch.nn.Sequential(model.stem, *model.blocks[:model.exit_layers[exit_index]+1],
                                                 model.exits[exit_index])
        else:
            combined_model = torch.nn.Sequential(model.stem, *model.blocks,
                                                 model.adaptive_pool, nn.Flatten(), model.final_classifier)

    elif name == "msdnet1":
        in_channels = 32  # Starting channels after the stem
        cumulative_blocks = []

        # Add stem and required blocks up to the exit index
        cumulative_blocks.append(model.stem)
        for i in range(exit_index + 1):
            cumulative_blocks.append(model.blocks[i])
            in_channels += model.growth_rate  # Track channels after each block

        if exit_index < len(model.exits):
            # Create the Conv2d layer in the exit to match cumulative channels
            exit_conv = nn.Conv2d(in_channels, model.exits[exit_index][0].out_channels, kernel_size=1)
            exit_layer = nn.Sequential(
                exit_conv,  # Replace Conv2d with dynamically calculated channels
                model.exits[exit_index][1],
                model.exits[exit_index][2]
            )
            cumulative_blocks.append(exit_layer)
            combined_model = nn.Sequential(*cumulative_blocks)
        else:
            # Final full model if exit_index exceeds available exits
            combined_model = nn.Sequential(
                model.stem,
                *model.blocks,
                model.adaptive_pool,
                nn.Flatten(),
                model.final_classifier
            )

    elif name == "resnet":
        if exit_index < len(model.exits):
            combined_model = nn.Sequential(
                *list(model._modules["model"])[:model.exit_layers[exit_index] + 1],
                model.exits[str(exit_index)]
            )
        else:
            combined_model = nn.Sequential(
                *list(model._modules["model"]),
                model.out_put
            )

    elif name == "mobilenet":
        if exit_index < len(model.exits):
            combined_model = nn.Sequential(
                *list(model._modules["model"][0])[:model.exit_layers[exit_index] + 1],
                model.exits[str(exit_index)]
            )
        else:
            combined_model = nn.Sequential(
                *list(model._modules["model"]),
                model.out_put
            )
    else:
        print("Invalid model name. Available models: 'mobilnet', 'resnet', and 'msdnet'")

    return combined_model




def evaluate_xai_on_exits(msdnet_model, testloader, device, background, num_classes=100):
    msdnet_model.to(device)
    msdnet_model.eval()

    all_scores = {i: [] for i in range(msdnet_model.num_exits)}
    all_attributions = {i: [] for i in range(msdnet_model.num_exits)}

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            for i in range(inputs.size(0)):
                sample = inputs[i:i + 1]
                target = labels[i].item()

                # Apply XAI methods to each exit
                for exit_index in range(msdnet_model.num_exits):
                    scores, attribution_maps = apply_xai_to_exit(msdnet_model, sample, background, target, exit_index)
                    # Store the scores and attribution maps for analysis
                    all_scores[exit_index].append(scores)
                    all_attributions[exit_index].append(attribution_maps)
                visualize_attributions(sample, attribution_maps, list(attribution_maps.keys()))

    return all_scores, all_attributions




def apply_xai_to_exit(model, sample, target, exit_index):
    # Initialize the combined model up to the specific exit
    combined_model = get_combined_model(model._name, model, exit_index)
    combined_model.cuda()
    # Initialize explainers
    integrated_gradients = IntegratedGradients(combined_model)
    saliency = Saliency(combined_model)
    smooth_grad = NoiseTunnel(saliency)
    occlusion = Occlusion(combined_model)

    # Grad-CAM using pytorch_grad_cam
    target_layer = model.get_target_layer(exit_index)
    grad_cam_plus_plus = GradCAMPlusPlus(model=combined_model, target_layers=[target_layer])

    # Apply Integrated Gradients
    try:
        ig_attributions = integrated_gradients.attribute(sample, target=target)
        ig_score = torch.mean(ig_attributions).item()
    except Exception as e:
        print("Integrated Gradients error:", e)
        ig_attributions, ig_score = None, None

    # Apply SmoothGrad
    try:
        smooth_grad_attributions = smooth_grad.attribute(sample, target=target, nt_type='smoothgrad', stdevs=0.2)
        smooth_grad_score = torch.mean(smooth_grad_attributions).item()
    except Exception as e:
        print("SmoothGrad error:", e)
        smooth_grad_attributions, smooth_grad_score = None, None

    # Apply Grad-CAM
    #try:
    grad_cam_target = [ClassifierOutputTarget(target)]

    grad_cam_attributions = grad_cam_plus_plus(input_tensor=sample, targets=grad_cam_target)
    grad_cam_score = grad_cam_attributions.mean()
    # except Exception as e:
    #     print("Grad-CAM error:", e)
    #     grad_cam_attributions, grad_cam_score = None, None
    #     # Try LRP with custom handling for unsupported layers
    occlusion_attributions = occlusion.attribute(
        sample,
        strides=(3, 8, 8),  # Stride for channels, height, width
        sliding_window_shapes=(3, 15, 15),  # Sliding window size
        target=target
    )
    occlusion_score = torch.mean(occlusion_attributions).item()

    # Prepare scores and attribution maps
    scores = {
        "IG": ig_score,
        "SmoothGrad": smooth_grad_score,
        "GradCam": grad_cam_score,
        "Occlusion": occlusion_score
    }

    attribution_maps = {
        "IG": ig_attributions.cpu().detach() if ig_attributions is not None else None,
        "SmoothGrad": smooth_grad_attributions.cpu().detach() if smooth_grad_attributions is not None else None,
        "GradCam": grad_cam_attributions if grad_cam_attributions is not None else None,
        "Occlusion": occlusion_attributions.cpu().detach()
    }
    return scores, attribution_maps


def conventional_xai_scores(basemodel, exit_layer, sample, background, target):

    combined_model = torch.nn.Sequential(basemodel, exit_layer)

    # Initialize explainers
    deep_lift_shap = DeepLiftShap(combined_model)
    saliency = Saliency(combined_model)
    integrated_gradients = IntegratedGradients(combined_model)
    grad_cam = LayerGradCam(combined_model, basemodel[-1])  # Target the last layer in basemodel for GradCam

    # DeepLift SHAP
    shap_attributions = deep_lift_shap.attribute(sample, background, target=target, return_convergence_delta=False)
    shap_score = torch.mean(shap_attributions).item()

    # Saliency
    saliency_attributions = saliency.attribute(sample, target=target)
    saliency_score = torch.mean(saliency_attributions).item()

    # Integrated Gradients
    ig_attributions = integrated_gradients.attribute(sample, target=target)
    ig_score = torch.mean(ig_attributions).item()

    # Grad-CAM
    grad_cam_attributions = grad_cam.attribute(sample, target=target)
    grad_cam_score = torch.mean(grad_cam_attributions).item()

    # Return both scores and full attribution maps
    scores = {
        "shap": shap_score,
        "saliency": saliency_score,
        "integrated_gradients": ig_score,
        "grad_cam": grad_cam_score
    }

    attribution_maps = {
        "shap": shap_attributions.cpu().detach(),
        "saliency": saliency_attributions.cpu().detach(),
        "integrated_gradients": ig_attributions.cpu().detach(),
        "grad_cam": grad_cam_attributions.cpu().detach()
    }

    return scores, attribution_maps



def getAttribution(model, input_image, default_thresholds=[0.4, 0.4, 0.4]):
    predicted_classes = []
    pfams_list = []
    exit_reached = False
    cumulative_map = None
    cumulative_maps=[]
    # Containers for storing scores and attribution maps for each exit
    all_scores = {i: [] for i in range(model.num_exits + 1)}
    all_attributions = {i: [] for i in range(model.num_exits + 1)}

    # Containers for cumulative maps of each XAI method
    xai_maps = {
        "IG": None,
        "SmoothGrad": None,
        "GradCam": None,
        "Occlusion": None
    }
    num_exits = model.num_exits + 1  # Total exits including final

    for exit_idx in range(model.num_exits):
        print("Processing exit:", exit_idx)

        # Forward pass to current exit
        output, class_idx, target_score = model.forward_to_exit(input_image, exit_idx,True)
        predicted_classes.append(class_idx)

        #_channel_weighted_maps_xai = getattr(model, '_xai_cwmaps', [])
        pfams_list, cumulative_map, _channel_weighted_maps_xai = generate_pfams(
            model.activations, model.gradients, input_image, pfams_list,
            _channel_weighted_maps_xai
        )
        model._xai_cwmaps = _channel_weighted_maps_xai
        _channel_weighted_maps_xai = getattr(model, '_xai_cwmaps', [])
        pfams_list, cumulative_map, _channel_weighted_maps_xai = generate_pfams(
            model.activations, model.gradients, input_image, pfams_list,
            _channel_weighted_maps_xai
        )
        model._xai_cwmaps = _channel_weighted_maps_xai
        cumulative_maps.append(cumulative_map)
        progressive_score = calculate_progressive_score(cumulative_maps)

        # Calculate IEES and confidence scores
        _Q_xai = calculate_consistency_index(cumulative_maps)
        _w_xai = [0.5, 0.3, 0.2]  # default weights for XAI visualization
        iees_score, confidence, gwtedAct, _Q_ret, activation_score, gradient_score = \
            compute_iees_score(model.activations, model.gradients, output, _Q_xai, _w_xai)

        threshold = default_thresholds[exit_idx] if exit_idx < len(default_thresholds) else 0.4
        exit_decision = iees_score > threshold

        # Apply XAI methods
        scores, attribution_maps = apply_xai_to_exit(model, input_image, predicted_classes[exit_idx], exit_idx)
        all_scores[exit_idx].append(scores)
        all_attributions[exit_idx].append(attribution_maps)

        # Aggregate cumulative maps for each XAI method
        for method in xai_maps.keys():
            method_map = attribution_maps[method]
            if xai_maps[method] is None:
                xai_maps[method] = method_map
            else:
                xai_maps[method] += method_map

        # Apply exit decision
        if exit_decision:
            print(f"Exiting at exit {exit_idx} with combined IEES score: {iees_score}")
            exit_reached = True
            break  # Early exit

    # If no early exit was made, process the final exit
    if not exit_reached:
        print("No early exit met criteria; processing final exit.")
        final_exit_idx = len(model.exits)  # Index for the final exit
        output, class_idx, target_score = model.forward_to_exit(input_image, final_exit_idx)
        predicted_classes.append(class_idx)

        _channel_weighted_maps_xai = getattr(model, '_xai_cwmaps', [])
        pfams_list, cumulative_map, _channel_weighted_maps_xai = generate_pfams(
            model.activations, model.gradients, input_image, pfams_list,
            _channel_weighted_maps_xai
        )
        model._xai_cwmaps = _channel_weighted_maps_xai
        #pfams_list, cumulative_map = compute_pfam_ig(model.activations, model.gradients, input_image, pfams_list)
        cumulative_maps.append(cumulative_map)
        progressive_score = calculate_progressive_score(cumulative_maps)

        _Q_xai = calculate_consistency_index(cumulative_maps)
        _w_xai = [0.5, 0.3, 0.2]  # default weights for XAI visualization
        iees_score, confidence, gwtedAct, _Q_ret, activation_score, gradient_score = \
            compute_iees_score(model.activations, model.gradients, output, _Q_xai, _w_xai)

        scores, attribution_maps = apply_xai_to_exit(model, input_image, predicted_classes[final_exit_idx],
                                                     final_exit_idx)
        all_scores[final_exit_idx].append(scores)
        all_attributions[final_exit_idx].append(attribution_maps)

        # Aggregate cumulative maps for each XAI method
        for method in xai_maps.keys():
            method_map = attribution_maps[method]
            if xai_maps[method] is None:
                xai_maps[method] = method_map
            else:
                xai_maps[method] += method_map

    # Normalize cumulative maps by the number of exits
    for method in xai_maps.keys():
        xai_maps[method] /= num_exits

    return pfams_list, cumulative_maps, xai_maps, predicted_classes, all_scores, all_attributions

# Function to compute progressive score based on cumulative map differences
def calculate_progressive_score(cumulative_maps):
    # Only calculate if there are at least two cumulative maps
    if len(cumulative_maps) > 1:
        return np.mean(np.abs(cumulative_maps[-1] - cumulative_maps[-2]))
    return 0  # Return 0 if there's no previous map for comparison

def normalize_tensor(tensor):
    """
    Normalize a tensor to have zero mean and unit variance.
    """
    mean = tensor.mean()
    std = tensor.std()
    normalized_tensor = (tensor - mean) / std if std != 0 else tensor
    return normalized_tensor


# def compute_iees_score(activations, gradients, output, progressive_score):
#     w_resnet = [0.6, 0.25, 0.15]
#     w=w_resnet
#     #W_msdnet= [0.8,  0.05, 0.15 ]
#     #W_mobilenetnet = [0.75, 0.1, 0.15]
#     #w=W_mobilenetnet
#
#     """
#     Compute the Interpretability-Based Early-Exit Score (IEES) for an exit.
#
#     Parameters:
#     - activations (torch.Tensor): Activations from the exit layer.
#     - gradients (torch.Tensor): Gradients from the exit layer.
#     - output (torch.Tensor): Model output at the exit.
#     - progressive_score (float or torch.Tensor): Progressive score for the model's progression.
#     - w (list of floats): Weights for A_iees, C_iees, and progressive score.
#
#     Returns:
#     - Tuple containing iees_score, C_iees, A_iees, activation_score, gradient_score, and normalized_progressive_score.
#     """
#
#     # Normalize activations and gradients
#     normalized_activations = normalize_tensor(activations)
#     normalized_gradients = normalize_tensor(gradients)
#
#     # Normalize the progressive score if it is a tensor
#     normalized_progressive_score = (
#         normalize_tensor(progressive_score) if isinstance(progressive_score, torch.Tensor) else progressive_score
#     )
#
#     # Attribution-based Component (A_iees)
#     attribution_map = normalized_activations * normalized_gradients
#     A_iees = attribution_map.abs().mean().item()
#
#     # Confidence-based Component (C_iees)
#     confidence_scores = F.softmax(output, dim=1)
#     C_iees, _ = confidence_scores.max(dim=1)
#     C_iees = C_iees.item()
#
#     # Combined IEES score
#     iees_score = w[0] * A_iees + w[1] * C_iees + w[2] * normalized_progressive_score
#
#     # Additional scores for interpretability analysis
#     activation_score = torch.mean(activations).item()
#     gradient_score = torch.mean(torch.abs(gradients)).item()
#
#     #print(f"IEEScore Breakdown - A_iees: {A_iees}, C_iees: {C_iees}, Prog_Score: {normalized_progressive_score}, Final IEEScore: {iees_score}")
#
#     return iees_score, C_iees, A_iees, activation_score, gradient_score, normalized_progressive_score

def compute_integrated_gradients(activations, gradients, input_tensor, baseline=None, steps=50):
    """
        Compute Integrated Gradients for a single exit layer using pre-collected activation and gradient.

        Parameters:
        - activation: Activation tensor at the specific exit layer.
        - gradient: Gradient tensor at the specific exit layer corresponding to the activation.
        - input_tensor: The original input tensor for which we compute attributions.
        - baseline: The baseline input to compare against. If None, uses a zero baseline.
        - steps: Number of steps in the Riemann sum approximation of the integral.

        Returns:
        - exit_wise_map: Attribution map for the specific exit layer.
        """

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

        # Generate scaled inputs between baseline and input_tensor
    scaled_inputs = [(baseline + float(i) / steps * (input_tensor - baseline)) for i in range(steps + 1)]

    # Initialize attribution map
    avg_gradients = torch.zeros_like(input_tensor)

    for scaled_input in scaled_inputs:
        print(f"scaled_input shape: {scaled_input.shape}")
        print(f"baseline shape: {baseline.shape}")
        print(f"gradients shape: {gradients.shape}")
        # Calculate the contribution of gradients for each scaled input
        scaled_gradients = (scaled_input - baseline) * gradients.mean(dim=(2, 3), keepdim=True)
        avg_gradients += scaled_gradients / steps

    # Compute exit-wise attribution based on the pre-collected activation and gradient
    exit_wise_map = avg_gradients * (input_tensor - baseline)

    return exit_wise_map.detach().cpu().numpy()

def compute_pfam_ig(model, input_tensor, target_class,pfams_list):
    exit_attributions = compute_integrated_gradients(model, input_tensor, target_class,  baseline=None, steps=50)
    pfams_list.append(exit_attributions)
    cumulative_map = np.mean(pfams_list, axis=0) if pfams_list else None
    cumulative_map1 = torch.stack(pfams_list).sum(dim=0)
    return pfams_list, cumulative_map

def getAttribution_qty(model, input_image):
    predicted_classes = []
    pfams_list = []
    all_attributions = {i: [] for i in range(model.num_exits + 1)}
    for exit_idx in range(model.num_exits+1):
        # Forward pass to current exit
        output, class_idx, target_score = model.forward_to_exit(input_image, exit_idx,True)
        predicted_classes.append(class_idx)
        _channel_weighted_maps_xai = getattr(model, '_xai_cwmaps', [])
        pfams_list, cumulative_map, _channel_weighted_maps_xai = generate_pfams(
            model.activations, model.gradients, input_image, pfams_list,
            _channel_weighted_maps_xai
        )
        model._xai_cwmaps = _channel_weighted_maps_xai
        scores, attribution_maps = apply_xai_to_exit(model, input_image, predicted_classes[exit_idx], exit_idx)
        attribution_maps["PFAM"]=pfams_list
        all_attributions[exit_idx].append(attribution_maps)
    return all_attributions