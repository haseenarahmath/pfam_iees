from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np

from src.grid import evaluate_confidence, evaluate_iees_proxy


def bayesian_optimize(model, dataloader, device, criteria_type='iees', calibration_mode='shared', n_calls=30):
    """
    Unified Bayesian Optimization for tuning exit policy parameters.

    Parameters:
    - model: Early-exit model
    - dataloader: Validation/Test DataLoader
    - device: 'cuda' or 'cpu'
    - criteria_type: 'iees' or 'confidence'
    - calibration_mode: 'shared' or 'per_exit'
    - n_calls: Number of BO optimization steps

    Returns:
    - best_config: Dictionary with optimal parameters
    """

    # Define parameter space and objective
    if criteria_type.lower() == 'iees':
        if calibration_mode == "shared":
            space = [
                Real(0.0, 1.0, name='alpha'),
                Real(0.0, 1.0, name='beta'),
                Real(0.0, 1.0, name='gamma'),
                Real(0.1, 0.95, name='threshold0'),
            ]

            @use_named_args(space)
            def objective(alpha, beta, gamma, threshold0):
                if abs(alpha + beta + gamma - 1.0) > 1e-3:
                    return 1.0  # Penalize invalid weight combinations
                acc = evaluate_iees_proxy(model, dataloader, device, calibration_mode,
                                    alpha, beta, gamma, threshold0)
                return -acc  # Minimize negative accuracy

        else:  # per-exit thresholds
            space = [
                Real(0.0, 1.0, name='alpha'),
                Real(0.0, 1.0, name='beta'),
                Real(0.0, 1.0, name='gamma'),
                Real(0.1, 0.95, name='threshold0'),
                Real(0.1, 0.95, name='threshold1'),
                Real(0.1, 0.95, name='threshold2'),
            ]

            @use_named_args(space)
            def objective(alpha, beta, gamma, threshold0, threshold1, threshold2):
                if abs(alpha + beta + gamma - 1.0) > 1e-3:
                    return 1.0
                acc = evaluate_iees_proxy(model, dataloader, device, calibration_mode,
                                    alpha, beta, gamma,
                                    threshold0, threshold1, threshold2)
                return -acc

    elif criteria_type.lower() == 'confidence':
        if calibration_mode == "shared":
            space = [Real(0.1, 0.95, name='threshold0')]

            @use_named_args(space)
            def objective(threshold0):
                acc = evaluate_confidence(model, dataloader, device, calibration_mode, threshold0)
                return -acc

        else:
            space = [
                Real(0.1, 0.95, name='threshold0'),
                Real(0.1, 0.95, name='threshold1'),
                Real(0.1, 0.95, name='threshold2'),
            ]

            @use_named_args(space)
            def objective(threshold0, threshold1, threshold2):
                acc = evaluate_confidence(model, dataloader, device, calibration_mode,
                                          threshold0, threshold1, threshold2)
                return -acc
    else:
        raise ValueError(f"Unsupported criteria type: {criteria_type}. Choose 'iees' or 'confidence'.")

    # Run Bayesian Optimization
    result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

    # Extract results
    best_config = dict(zip([dim.name for dim in space], result.x))
    best_acc = -result.fun

    # Print results
    print(f"[BO - {criteria_type.upper()}] Best Accuracy: {best_acc:.4f}")
    print(f"[BO - {criteria_type.upper()}] Best Config:", best_config)

    return best_config
