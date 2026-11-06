# src/models/msdnet_imagenet.py

import torch
import argparse

from src.msdnet import MSDNet


class MSDNetImageNet(MSDNet):
    """
    ImageNet MSDNet.
    Inherits everything from kalviny MSDNet.
    Just sets correct args and loads weights.
    """

    def __init__(self, num_classes=1000,
                 num_exits=5):

        args = argparse.Namespace(
            nBlocks=num_exits,
            nChannels=32,
            base=4,
            stepmode='even',
            step=4,
            growthRate=16,
            grFactor=[1, 2, 4, 4],
            bnFactor=[1, 2, 4, 4],
            nScales=4,
            reduction=0.5,
            bottleneck=True,
            priorNet=True,
            data='ImageNet',
            prune='max')

        super().__init__(args)
        self._name = 'msdnet_imagenet'

    def load_pretrained(self, path):
        ckpt = torch.load(
            path,
            map_location='cpu',
            weights_only=False)

        # Strip DataParallel prefix
        state = {
            k.replace('module.', ''): v
            for k, v in
            ckpt['state_dict'].items()}

        missing, unexpected = \
            self.load_state_dict(
                state, strict=False)

        if not missing and not unexpected:
            print("✅ Perfect weight match!")
        else:
            print(f"Missing:    {len(missing)}")
            print(f"Unexpected: {len(unexpected)}")

        acc = 100 - ckpt['best_err1']
        print(f"✅ Best acc: {acc:.2f}%")
        return acc