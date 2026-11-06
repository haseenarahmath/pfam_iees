# IEES + PFAM Framework

This repository provides code for our framework that combines:

- **Interpretability-Based Early-Exit Score (IEES)** with a proxy predictor for efficient inference.  
- **Progressive Feature Attribution Maps (PFAMs)** for visual transparency.  

It includes scripts for training, inference, robustness tests, ablations, and quantitative evaluation.

---
## Usage

The main entry point is `main.py`. Select the operation with `--operation`.

### Key Arguments
- `--model`: Backbone (`resnet`, `msdnet`, `mobilenet`)  
- `--exit_criterion`: Exit policy (`iees`, `confidence`)  
- `--operation`: One of the supported operations (see below)  
- `--data_dir`: Path to dataset  
- `--output_dir`: Directory for results  
- `--cur_type`: Corruption type for `drift` (e.g., `gaussian_noise`, `brightness`, `motion_blur`)  
- `--deployment`: Set `True` to disable PFAM generation for fast inference  
- `--visuals`: Set `True` to generate PFAM visuals during inference 

---

## Supported Operations

| Operation        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `proxy_training` | Train the Random Forest proxy predictor for IEES                           |
| `inference`      | Run proxy-based inference; analysis mode can optionally generate PFAM visualizations |
| `comparison_xai` | Compare PFAM with XAI baselines (Grad-CAM, IG, SmoothGrad, Occlusion)      |
| `bbc_plot`       | Plot Budgeted Batch Classification (BBC) curves                            |
| `budget`         | Budget-controlled evaluation (FLOPs–accuracy trade-offs)                   |
| `drift`          | Distribution shift evaluation on CIFAR-10-C (`--cur_type` required)        |
| `quantitative`   | Quantitative PFAM evaluation (semantic progression, MFA, convergence IoU, faithfulness AUCs) |
| `ablation`       | IEES ablation (conf-only, +prog, +rel, w/o variants)                       |

---

## Notes

- Dataset selection is automatic:  
  - ResNet → CIFAR-10  
  - MSDNet → CIFAR-100  
  - MobileNet → ImageNet  

- **Datasets are not bundled with this build.**  
- CIFAR-10 and CIFAR-100 will be downloaded automatically when running experiments.  
- ImageNet is **not automatically downloaded**:  
  - Place the full ImageNet dataset inside the `data/` folder if using MobileNet.  
  - A **small subset of ImageNet** is included with this build for quick testing and sanity checks.  

- Drift evaluation is supported only for ResNet on CIFAR-10-C.  

## Example Runs

Set environment (optional):  
```bash
export CUDA_VISIBLE_DEVICES=0
```


Run inference with the trained proxy:  
```bash
python main.py   --operation inference   --model resnet   --exit_criterion iees   --output_dir results/   --data_dir data
```

Compare PFAM with XAI baselines:  
```bash
python main.py   --operation comparison_xai   --model resnet   --exit_criterion iees   --output_dir results/   --data_dir data
```

Train the proxy predictor:  
```bash
python main.py   --operation proxy_training   --model resnet   --exit_criterion iees   --output_dir results/   --data_dir data   --num_exits 3   --num_epochs 200
```


Plot Budgeted Batch Classification (BBC):  
```bash
python main.py   --operation bbc_plot   --model resnet   --output_dir results/   --data_dir data
```

Quantitative PFAM evaluation:  
```bash
python main.py   --operation quantitative   --model resnet   --exit_criterion iees   --output_dir results/   --data_dir data
```

Distribution drift evaluation (ResNet + CIFAR-10-C):  
```bash
python main.py   --operation drift   --model resnet   --exit_criterion iees   --output_dir results/   --data_dir data   --cur_type gaussian_noise
```

IEES ablation (runs Conf-only, +Prog, +Rel, and w/o variants):  
```bash
python main.py   --operation ablation   --model resnet   --exit_criterion iees   --output_dir results/   --data_dir data
```

---

## Outputs

Results are stored in `results/`. Key outputs include:

- `csv/proxy_*`: Proxy training metrics  
- `csv/inference_*`: Inference results  
- `csv/comparison_xai_*`: XAI baseline comparisons  
- `csv/bbc_*`: BBC evaluation results  
- `csv/grid_*`, `csv/bo_*`: Calibration results (grid / BO)  
- `csv/budget_*`: Budget evaluations  
- `csv/drift_*`: Drift results (CIFAR-10-C)  
- `csv/pfam_quantitative_*`: PFAM quantitative metrics  
- `csv/ablation_*`: Ablation results  

Figures (PFAM visualizations, BBC plots, ablation curves) are saved under `results/` in corresponding subfolders.

---
