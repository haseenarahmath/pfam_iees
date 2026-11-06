import json

import torch
import torch.nn as nn
import csv

from sklearn.metrics import auc

from model import MultiExitResNet18, MultiExitResNet50, MSDNet, MultiExitMobileNetV3

def load_model_config_json(config_file="./model_config.json"):
    with open(config_file, 'r') as f:
        return json.load(f)

def get_details(m_name, config_file="./model_config.json"):
    config = load_model_config_json(config_file)
    if m_name not in config:
        raise ValueError(f"Unknown model: {m_name}")
    return config[m_name]['dataset'],config[m_name]['numclasses'],config[m_name]['optimal_threshold'],config[m_name]['optimal_threshold_conf'],config[m_name]['weights']

def load_model(m_name,num_classes,num_exits=3):
    print("Loading model ",m_name)
    if m_name=="resnet":
        model= MultiExitResNet18(num_classes,num_exits)
    if m_name == "resnet50":
        model=  MultiExitResNet50(num_classes,num_exits)
    if m_name == "mobilenet":
        model=  MultiExitMobileNetV3(num_classes,num_exits)
    if m_name == "msdnet":
        model=  MSDNet(num_classes,num_exits)
    model._name = m_name
    return model


def getCumulativeFlops(model_name, baseline=False, config_file="./model_config.json"):
    config = load_model_config_json(config_file)
    if model_name not in config:
        raise ValueError(f"Unknown model: {model_name}")

    model_data = config[model_name]
    if baseline:
        return model_data['baseline_flops'][0]  # Assuming the first entry is the baseline
    else:
        return model_data['flops']

def get_bck_acc(model_name, config_file="./model_config.json"):
    config = load_model_config_json(config_file)
    if model_name not in config:
        raise ValueError(f"Unknown model: {model_name}")
    return config[model_name]['backbone_accuracy']

def norm_flops(model_name,flp):
    unit = 'G' if model_name.lower() == 'msdnet' else 'M'
    factor = 1e9 if unit == 'G' else 1e6
    flops = flp / factor
    return  flops,unit

# Get the optimal threshold for a given model from the JSON configuration
def get_optimal_threshold(model_name, config_file="./model_config.json"):
    config = load_model_config_json(config_file)
    model=model_name.split("_")[0]
    if model not in config:
        raise ValueError(f"Unknown model: {model_name}")
    if "conf" in model_name:
        return config[model].get('optimal_threshold_conf', None)
    else:
        return config[model].get('optimal_threshold', None)

def get_optimal_global_threshold(model_name, config_file="./model_config.json"):
    config = load_model_config_json(config_file)
    model=model_name.split("_")[0]
    if model not in config:
        raise ValueError(f"Unknown model: {model_name}")
    if "conf" in model_name:
        return config[model].get('optimal_global_threshold_conf', None)
    else:
        return config[model].get('optimal_global_threshold', None)

def normalize_flops(model_name, bbc_df):
    flops,unit = norm_flops(model_name,bbc_df['avg_flops'])
    acc = bbc_df['avg_accuracy'] * 100  # convert to %
    return flops, acc, unit



def compute_auc(bbc_df):
    norm_flops = bbc_df['avg_flops'] / bbc_df['avg_flops'].max()
    norm_acc = bbc_df['avg_accuracy']  # already in [0–1]
    return auc(norm_flops, norm_acc)

def save_model_params(model_save_path,epoch,model,optimizer,scheduler,loss):
    # After training
    torch.save({
        'epoch': epoch,                # Last epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state if using one
        'loss': loss                   # Last training loss
    }, model_save_path)

def save_model(model_save_path, epoch, model, optimizer, scheduler, loss):
    # After training
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state if using one
        'loss': loss  # Last training loss
    }, model_save_path)

def reorganize_attribution_maps(attribution_data):
    # Find the first non-empty entry in attribution_data to initialize keys dynamically
    first_non_empty = next((method_dict_list[0] for method_dict_list in attribution_data.values() if method_dict_list),
                           None)

    if first_non_empty is None:
        raise ValueError("Attribution data is empty or all entries are empty.")

    # Initialize reorganized_data with keys from the first non-empty entry
    reorganized_data = {method: [] for method in first_non_empty.keys()}

    # Populate reorganized_data with attribution data from each exit
    for exit_index, method_dict_list in attribution_data.items():
        if method_dict_list:
            method_dict = method_dict_list[0]
            for method, tensor in method_dict.items():
                reorganized_data[method].append(tensor)

    return reorganized_data



def replace_inplace_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            # Replace the in-place ReLU with a non-in-place version
            setattr(model, child_name, nn.ReLU(inplace=False))
        elif isinstance(child, nn.Sequential) or isinstance(child, nn.Module):
            # Recursively apply this function to nested models
            replace_inplace_relu(child)

def write_into_csv(sample,exit_idx, iees_score, confidence, progressive_score,gwtedAct, activation_score, gradient_score, pred, label,decision, exit, filename="results/csv/output.csv"):
    # Define the header and the row data
    header = ["Sample","Exit Index", "IEEScore", "Confidence", "Consistency Index","Weighted Activation", "Activation Score",
              "Gradient Score", "Predicted Label", "Actual Label", "Descision","Prediction Accuracy"]
    row = [sample,exit_idx, iees_score, confidence, progressive_score,gwtedAct, activation_score, gradient_score, pred, label,decision, exit]

    # Write to CSV file
    try:
        # Open the file in append mode
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            # Write the header only if the file is empty
            file.seek(0, 2)  # Go to end of file
            if file.tell() == 0:
                writer.writerow(header)
            writer.writerow(row)
        print("Data written successfully.")
    except Exception as e:
        print(f"An error occurred while writing to CSV: {e}")


def save_to_csv(bbc_results,exit_type,fn=None):
    if fn is None:
        csv_filename = f'../results/csvgen/bbc_results_{exit_type}.csv'
    else:
        csv_filename=fn

    # Write the list of dictionaries to the CSV file
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=bbc_results.keys())

        # Write the header
        writer.writeheader()

        # Write the data rows
        writer.writerows([bbc_results])

    print(f"Data has been saved to {csv_filename}")

def generate_thresholds():
    # Helper function to generate a range of values with a given step
    def frange(start, stop, step):
        while start <= stop + 1e-8:  # Avoid floating-point precision issues
            yield start
            start += step

    # Coarse thresholds outside the informative region
    coarse_low = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    coarse_high = [0.91, 0.95, 0.99]

    # Fine-grained thresholds in the informative region
    fine = [round(val, 2) for val in frange(0.4, 0.9, 0.01)]

    # Combine and return the sorted list of thresholds (no duplicates)
    thresholds = coarse_low + fine + coarse_high
    return thresholds