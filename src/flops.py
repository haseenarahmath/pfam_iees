import argparse
import torch
from ptflops import get_model_complexity_info
#pip install ptflops==0.6.9

from src.model import MultiExitResNet18, MultiExitMobileNetV3, MSDNet


def get_model(name, dataset, num_classes):
    if name.lower() == "resnet":
        return MultiExitResNet18(num_classes=num_classes)
    elif name.lower() == "mobilenet":
        return MultiExitMobileNetV3(num_classes=num_classes)
    elif name.lower() == "msdnet":
        return MSDNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")

@torch.no_grad()
def compute_flops(model, input_shape):
    model.eval()
    flops = []
    for exit_idx in range(model.num_exits):
        def forward_hook(x):
            return model.forward_to_exit(x, exit_idx)[0]
        macs, _ = get_model_complexity_info(model, input_shape, as_strings=False,
                                            print_per_layer_stat=False,
                                            input_constructor=lambda _: torch.randn(1, *input_shape),
                                            custom_forward=forward_hook)
        flops.append(macs)
    # Baseline full exit
    macs, _ = get_model_complexity_info(model, input_shape, as_strings=False,
                                        print_per_layer_stat=False,
                                        input_constructor=lambda _: torch.randn(1, *input_shape),
                                        custom_forward=lambda m: m.forward_to_exit(torch.randn(1, *input_shape), model.num_exits)[0])
    flops.append(macs)
    return flops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "mobilenet", "msdnet"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet"])
    args = parser.parse_args()

    # Dataset-specific settings
    shape = {
        "cifar10": (3, 32, 32),
        "cifar100": (3, 32, 32),
        "imagenet": (3, 224, 224)
    }[args.dataset]

    num_classes = {
        "cifar10": 10,
        "cifar100": 100,
        "imagenet": 1000
    }[args.dataset]

    model = get_model(args.model, args.dataset, num_classes)
    flops = compute_flops(model, shape)

    for idx, mac in enumerate(flops):
        print(f"Exit {idx}: {mac} FLOPs")

if __name__ == "__main__":
    main()
