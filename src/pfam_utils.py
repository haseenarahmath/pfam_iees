from src.iees_utils import (
    generate_pfams,
    DEFAULT_COMMON_RESOLUTION,
)


class ExitAttributionCache:
    """
    Lightweight cache that accumulates per-exit data during a forward pass.

    Attributes
    ----------
    activations : dict  {exit_idx → torch.Tensor (1, K, H, W)}
        Feature maps Z_{i,k}(x) at each exit, detached from the graph.
    outputs     : dict  {exit_idx → torch.Tensor (1, C)}
        Raw logits produced by each exit head.
    class_ids   : dict  {exit_idx → int}
        Predicted class index (argmax of softmax) at each exit.
    """

    def __init__(self):
        self.activations = {}
        self.outputs     = {}
        self.class_ids   = {}

    def cache(self, exit_idx, model, output, class_idx):
        """
        Store activations, output logits, and predicted class for one exit.

        Parameters
        ----------
        exit_idx  : int   — zero-based exit index
        model     : nn.Module with a `.activations` attribute (torch.Tensor)
                    set by a forward hook at the target conv layer.
        output    : torch.Tensor  (1, C)  — exit-head logits (requires_grad)
        class_idx : torch.Tensor  scalar  — argmax predicted class
        """
        self.outputs[exit_idx]     = output.clone()
        self.class_ids[exit_idx]   = class_idx.item()
        self.activations[exit_idx] = model.activations.detach().clone()



def generate_pfams_after_exit(
    model,
    input_tensor,
    final_exit_idx,
    activation_cache,
    output_cache,
    class_cache,
    common_resolution=DEFAULT_COMMON_RESOLUTION,
):
    """
    Generate exit-wise PFAM maps and cumulative maps C_i(x)  (paper Eq. 2–4)
    using cached activations and outputs.

    All attribution maps are upsampled to `common_resolution` (H0×W0) before
    accumulation, ensuring spatial compatibility across exits  (paper Eq. 3).

    Parameters
    ----------
    model            : nn.Module — must expose `.gradients` after backward
    input_tensor     : torch.Tensor  (1, C_in, H_in, W_in)
    final_exit_idx   : int  — last exit index to process (inclusive, 0-based)
    activation_cache : dict  {exit_idx → torch.Tensor (1, K, H, W)}
    output_cache     : dict  {exit_idx → torch.Tensor (1, num_classes)}
    class_cache      : dict  {exit_idx → int}
    common_resolution: tuple (H0, W0)

    Returns
    -------
    pfams_list            : list of np.ndarray  — A_1(x), …, A_i(x)
                            each shape (H0, W0), normalised to [0, 1]
    cumulative_maps       : list of np.ndarray  — C_1(x), …, C_i(x)
                            each shape (H0, W0)
    channel_weighted_maps : list of np.ndarray  — M_1(x), …, M_i(x)
                            each shape (H0, W0), raw (needed for F_i(x))
    """
    pfams_list             = []
    cumulative_maps        = []
    channel_weighted_maps  = []

    for i in range(final_exit_idx + 1):
        model.zero_grad()

        # --- Backpropagate class score to get gradients at exit i ---
        class_score = output_cache[i][0, class_cache[i]]
        class_score.backward(retain_graph=True)

        gradients   = model.gradients.detach()          # (1, K, H, W)
        activations = activation_cache[i]               # (1, K, H, W)

        # --- generate_pfams handles Eq. 2 (Grad-CAM) and Eq. 3 (cumulative) ---
        pfams_list, cumulative_map, channel_weighted_maps = generate_pfams(
            activations,
            gradients,
            input_tensor,
            pfams_list,
            channel_weighted_maps,
            common_resolution=common_resolution,
        )
        cumulative_maps.append(cumulative_map)

    return pfams_list, cumulative_maps, channel_weighted_maps



def generate_pfams_after_exit1(
    model,
    input_tensor,
    n_exits,
    common_resolution=DEFAULT_COMMON_RESOLUTION,
):
    """
    Generate PFAM maps by running a forward pass to each exit in sequence.

    Suitable when cached activations are not pre-stored.  The model must
    implement `forward_to_exit(input_tensor, exit_idx, compute_grad=True)`
    and expose `.activations` and `.gradients` after the call.

    Parameters
    ----------
    model            : nn.Module with forward_to_exit, .activations, .gradients
    input_tensor     : torch.Tensor  (1, C_in, H_in, W_in)
    n_exits          : int  — total number of exits
    common_resolution: tuple (H0, W0)

    Returns
    -------
    pfams_list     : list of np.ndarray  — A_1(x), …, A_{n}(x)
    cumulative_maps: list of np.ndarray  — C_1(x), …, C_{n}(x)
    channel_weighted_maps : list of np.ndarray  — M_1(x), …, M_{n}(x)
    """
    pfams_list             = []
    cumulative_maps        = []
    channel_weighted_maps  = []

    for exit_idx in range(n_exits):
        # forward_to_exit must trigger backward for the predicted class score
        # and store gradients in model.gradients, activations in model.activations
        _output, _class_idx, _target_score = model.forward_to_exit(
            input_tensor, exit_idx, True
        )

        pfams_list, cumulative_map, channel_weighted_maps = generate_pfams(
            model.activations,
            model.gradients,
            input_tensor,
            pfams_list,
            channel_weighted_maps,
            common_resolution=common_resolution,
        )
        cumulative_maps.append(cumulative_map)

    return pfams_list, cumulative_maps, channel_weighted_maps
