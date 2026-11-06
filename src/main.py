import argparse
import os
import joblib
import torch
from datasets import load_data, load_corrupted_cifar
from ablation import run_iees_ablation
from bo import bayesian_optimize
from eval_core import evaluate_core
from grid import grid_search_iees
from proxy import train_proxy
from visuals import plot_bbc
from train import load_and_train_model, inference, comparision_with_XAI_tools, distribution_drift, \
    evaluate_model_thresholds
from util import get_details, save_to_csv


def main(args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset, numclasses, iees_threshold, conf_threshold, weights = get_details(args.model)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    optimal_threshold= iees_threshold
    train_loader, val_loader, test_loader = load_data(dataset, size=0, data_dir=args.data_dir)

    # Step 1: Train model
    model = load_and_train_model(
        args.model, numclasses, args.num_exits, device, args.deployment, args.num_epochs,
        train_loader, val_loader
    )
    model.eval()

    # Step 2: Train proxy
    if args.operation == "proxy_training":
        regressor = train_proxy(model, train_loader, val_loader, weights, device=device)#
        joblib.dump(regressor, f'../models/iees_proxy_rf_{dataset}.pkl')
    if args.operation == "inference":
        proxy_path = f'../models/iees_proxy_rf_{dataset}.pkl'
        if not os.path.exists(proxy_path):
            raise FileNotFoundError(
                f"Proxy model not found: {proxy_path}\n"
                f"Please run proxy training first using:\n"
                f"python main.py --operation proxy_training --model {args.model}"
            )
        regressor = joblib.load(proxy_path)
        inference(model, regressor,test_loader, device, optimal_threshold,dataset, output_dir, weights,generate_visuals=args.visuals)
    elif args.operation == "comparison_xai":
        comparision_with_XAI_tools(model, dataset, test_loader, device, optimal_threshold,weights)
    elif args.operation == "bbc_plot":
        iees_csv = f'{output_dir}/csv/bbc_results_IEEScore_{dataset}.csv'
        conf_csv = f'{output_dir}/csv/bbc_results_Confidence_{dataset}.csv'
        plot_bbc(args.model, iees_csv, conf_csv)
    elif args.operation == "drift":
        assert args.model == "resnet", \
            f"[ERROR] Drift evaluation is only supported for the 'resnet' model trained on CIFAR-10. The selected model is '{args.model}'."
        testloader=load_corrupted_cifar(args.cur_type)
        regressor = joblib.load(f'../models/iees_proxy_rf_{dataset}.pkl') if args.exit_criterion == "iees" else None
        bbc_results = distribution_drift(model, args.model, testloader, device, args.exit_criterion, optimal_threshold, weights, proxy=regressor)
        fn = f'{output_dir}/csv/correcpted_cifar_{args.cur_type}_{args.exit_criterion}.csv'
        save_to_csv(bbc_results, None, fn)
    elif args.operation == "budget":
        regressor = joblib.load(f'../models/iees_proxy_rf_{dataset}.pkl') if args.exit_criterion == "iees" else None
        bbc_results = evaluate_model_thresholds(model, args.model, test_loader, device, args.exit_criterion, optimal_threshold, weights, proxy=regressor)
        save_to_csv(bbc_results[0], args.exit_criterion + "_" + dataset)
    elif args.operation == "grid":
        regressor = joblib.load(f'../models/iees_proxy_rf_{dataset}.pkl')
        grid_search_iees(model, val_loader, device=device, rf_proxy=regressor, calibration_mode="shared")
    elif args.operation == "bo":
        bayesian_optimize(model, test_loader, device, criteria_type='iees', calibration_mode='per_exit', n_calls=30)
    elif args.operation == "quantitative":
        _ = evaluate_core(model,test_loader,device,max_samples=2,include_pfam=True,topk=0.20,normalize_mode="joint",
                          mass_norm=True, steps=40,baseline="mean",relu=True, gaussian_ksize=3,superpixel=True,
                          n_segments=200, compactness=10.0,temp=2.0, faithfulness_only_on_correct=True,force_full=True)
    elif args.operation == "ablation":
        out_csv = f"{args.output_dir}/csv/ablation_{args.model}_{dataset}.csv"
        df = run_iees_ablation(model=model,dataloader=test_loader,device=device,out_csv=out_csv,taus=optimal_threshold,
                               base_weights=weights,model_name=args.model)

        try: print(df.round(4))
        except Exception: pass
        print(f"[INFO] Ablation saved to {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the neural network operation.")

    # General parameters
    parser.add_argument('--output_dir', type=str, default="../results/", help="Directory for saving results")
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to dataset root directory')
    #
    parser.add_argument('--model', type=str, default='resnet', help='["resnet", "msdnet", "mobilenet"]')
    parser.add_argument('--exit_criterion', type=str, default="iees", help="Exit criterion (e.g., confidence, iees)")
    parser.add_argument('--operation', type=str, default="inference",
                        choices=["proxy_training", "inference", "comparison_xai", "bbc_plot", "drift", "budget", "grid", "bo", "ablation", "quantitative"], help="Type of operation to perform")
    parser.add_argument('--cur_type', type=str, default="gaussian_noise", help='["motion blur","brightness","gaussian_noise"]')

    parser.add_argument('--deployment', type=bool, default=True, help="Deployment flag")
    parser.add_argument('--visuals', type=bool, default=True, help="Generate  Visuals during inference")
    parser.add_argument('--gpu_index', type=int, default=0, help="GPU index for computation")
    parser.add_argument('--num_exits', type=int, default=3, help="Number of exits in the model")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs for training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
