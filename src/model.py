import torch.nn as nn
import torchvision.models as models

class MultiExitBase(nn.Module):
    def __init__(self, model, layer_channels, exit_layers, num_classes=10, out_put=None,num_exits=3):
        super(MultiExitBase, self).__init__()
        self.model = model
        self.num_exits = num_exits
        self.exit_layers = exit_layers
        self.layer_channels = layer_channels
        self.num_classes = num_classes
        self.out_put = out_put
        self.activations = None  # To store activations for each exit
        self.gradients = None  # To store gradients for each exit

        self.exits = nn.ModuleDict()
        for i, layer_no in enumerate(exit_layers):
            self.exits[str(i)] = self.create_exit(layer_channels[i], num_classes)

    def create_exit(self, in_channels, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        outputs = []
        i=-1
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in str(self.exit_layers):
                i += 1
                outputs.append(self.exits[str(i)](x))
        outputs.append(self.out_put(x))
        return outputs

    def get_target_layer(self, exit_index):
         # Return the first Conv layer in the exit block
        if exit_index < len(self.exits):
            return self.exits[str(exit_index)][0]   # Return the first Conv layer in the exit block
        else:
            return self.get_last_block_first_conv()

    def get_last_block_first_conv(self):
        main_model = self.model[-1]
        last_block = main_model[-1]
        if isinstance(last_block, nn.Sequential) or isinstance(last_block, nn.ModuleList):
            last_sub_block = last_block[-1]
            for layer in last_sub_block.children():
                if isinstance(layer, nn.Conv2d):
                    return layer
        else:
            if isinstance(last_block, nn.Conv2d):
                return last_block
            for layer in last_block.children():
                if isinstance(layer, nn.Conv2d):
                    return layer

        return None


    def save_gradients(self,grad):
        self.gradients = grad

    def forward_to_exit(self, x, exit_index,gradient_required=False):
        x.requires_grad_()  # Ensure gradients can be computed
        outputs = None
        final_layer = True if exit_index == len(self.exits) else False
        out_layer = self.out_put if final_layer else self.exits[str(exit_index)]
        for layer_id, layer in self.model._modules.items():
            x = layer(x)  # Propagate input through the layer

            if final_layer and layer_id == str(len(self.model._modules) - 2):
                if isinstance(layer, torch.nn.Sequential) and hasattr(layer[0], 'conv1'):
                    self.activations = x.clone()
                    x.register_hook(self.save_gradients)


            if (exit_index != len(self.exits) and layer_id == str(self.exit_layers[exit_index])) or \
                    (final_layer and layer_id == str(len(self.model._modules) - 1)):
                x = out_layer[0](x)
                if not final_layer:
                    self.activations = x.clone()  # Save the activations for Grad-CAM
                    x.register_hook(self.save_gradients)

                # Process through the remaining layers in the exit block
                for sub_layer in out_layer[1:]:
                    x = sub_layer(x)

                outputs = x
                break  # Stop after reaching the specified exit layer

        class_idx = outputs.argmax(dim=1)
        target_score = outputs[0, class_idx]  # Select the target class score for backpropagation
        if gradient_required:
            target_score.backward()
        return outputs, class_idx, target_score


class MultiExitResNet18(MultiExitBase):
    def __init__(self, num_classes=10,num_exits=3):
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base_model = nn.Sequential(*list(base_model.children())[:-2])
        layer_channels = [64, 128, 256]
        exit_layers = [4, 5, 6]
        out_put = nn.Sequential(
            #nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        super().__init__(base_model, layer_channels, exit_layers, num_classes, out_put,num_exits)


class MultiExitResNet50(MultiExitBase):
    def __init__(self, num_classes=10,num_exits=3):
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        base_model = nn.Sequential(*list(base_model.children())[:-2])
        layer_channels = [256, 512, 2048]
        exit_layers = ['layer1', 'layer2', 'layer4']
        out_put = nn.Sequential(
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        super().__init__(base_model, layer_channels, exit_layers, num_classes,out_put,num_exits)


class MultiExitMobileNetV3(MultiExitBase):
    def __init__(self, num_classes=10,num_exits=3):
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        base_model = nn.Sequential(*list(base_model.children())[:-2])
        layer_channels = [24, 40, 96]
        exit_layers = [2,6,9]

        output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),  # Flatten to make it [batch_size, 576]
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=1000, bias=True)
        )
        super().__init__(base_model, layer_channels, exit_layers, num_classes, output_layer, num_exits)

    def get_target_layer(self, exit_index):
         # Return the first Conv layer in the exit block
        if exit_index < len(self.exits):
            return self.exits[str(exit_index)][0]   # Return the first Conv layer in the exit block
        else:
            main_model = self.model._modules['0'][-1]
            return main_model[0]


    def forward(self, x):
        outputs = []
        i=-1
        for layer_id, layer in enumerate(self.model[0]):
            x = layer(x)
            if layer_id in self.exit_layers:
                i += 1
                outputs.append(self.exits[str(i)](x))
        outputs.append(self.out_put(x))
        return outputs

    def forward_to_exit(self, x, exit_index,gradient_required=False):
        x.requires_grad_()  # Ensure gradients can be computed

        outputs = None
        final_layer = True if exit_index == len(self.exits) else False
        out_layer = self.out_put if final_layer else self.exits[str(exit_index)]
        for layer_id, layer in enumerate(self.model[0]):
            # Propagate input through each top-level layer in the model
            x = layer(x)
            # Check if we're processing the final layer and register hooks on it
            if final_layer and layer_id == str(len(self.model._modules) - 2):
                if isinstance(layer, torch.nn.Sequential):
                    self.activations = x.clone()
                    x.register_hook(self.save_gradients)


            # Check if we have reached the specified exit layer
            if (exit_index != len(self.exits) and layer_id == self.exit_layers[exit_index]) or \
                    (final_layer and layer_id == len(self.model[0]) - 1):
                # Process through the layers in the exit block
                x = out_layer[0](x)
                if not final_layer:
                    # Store activations and set up the gradient hook
                    self.activations = x.clone()
                    x.register_hook(self.save_gradients)

                # Pass through the rest of the exit layers
                for sub_layer in out_layer[1:]:
                    x = sub_layer(x)

                outputs = x
                break  # Stop after reaching the specified exit layer

        class_idx = outputs.argmax(dim=1)
        target_score = outputs[0, class_idx]  # Select the target class score for backpropagation
        if gradient_required:
            target_score.backward()
        return outputs, class_idx, target_score


import torch
import torch.nn as nn
import torch.nn.functional as F

class MSDNetBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(MSDNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(growth_rate)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return torch.cat([x, out], dim=1)  # Concatenate input with output for dense connections

class MSDNet(nn.Module):
    def __init__(self, num_classes=10, num_exits=3, num_block=12, growth_rate=16):
        super(MSDNet, self).__init__()
        self.num_exits = num_exits
        self.num_block = num_block
        self.growth_rate = growth_rate
        self.exit_layers = [3, 5, 9]  # Ensure these are within block indices

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # Define blocks with early exits
        self.blocks = nn.ModuleList()
        self.exits = nn.ModuleList()
        in_channels = 32
        for i in range(num_block):
            # Each block increases channels by growth rate
            self.blocks.append(MSDNetBlock(in_channels, self.growth_rate))
            in_channels += self.growth_rate  # Update in_channels after concatenation

            # Add exit layer if this block is in exit_layers
            if i in self.exit_layers:
                # Ensure the exit Conv2d layer has input channels matching the current block's output
                self.exits.append(nn.Sequential(

                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, num_classes)
                ))

        # Final classifier after all blocks with adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.final_classifier = nn.Linear(in_channels, num_classes)

    def save_gradients(self, grad):
        self.gradients = grad

    def get_target_layer(self, exit_index):
        if exit_index < len(self.exits):
            exit_layer = self.exits[exit_index]
            for layer in exit_layer:
                if isinstance(layer, nn.Conv2d):
                    return layer
            raise ValueError("No Conv2d layer found in exit block")
        else:
            final_block = self.blocks[-1]
            for name, layer in final_block.named_children():
                if name == 'conv2':
                    return layer
            raise ValueError("No conv2 layer found in the final block")

    def forward(self, x, early_exit_threshold=None):
        outputs = []
        x = self.stem(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.exit_layers:
                exit_output = self.exits[self.exit_layers.index(i)](x)
                outputs.append(exit_output)

                # Early exit condition
                if early_exit_threshold:
                    confidence = F.softmax(exit_output, dim=1).max()
                    if confidence >= early_exit_threshold[self.exit_layers.index(i)]:
                        return exit_output, outputs

        # Final exit if no early exit is taken
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        final_output = self.final_classifier(x)
        outputs.append(final_output)

        return outputs

    def forward_to_exit(self, x, exit_index, gradient_required=False):
        x.requires_grad_()  # Enable gradients
        final_layer = exit_index == len(self.exits)
        outputs = None

        # Initial stem layer
        x = self.stem(x)
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Check if we are at the desired exit layer
            if not final_layer and i == self.exit_layers[exit_index] :
                for name, layer in self.exits[exit_index].named_children():
                    x = layer(x)
                    if isinstance(layer, nn.Conv2d):
                        self.activations = x.clone()
                        x.register_hook(self.save_gradients)
                outputs = x
                break

        if final_layer:
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
            outputs = self.final_classifier(x)

        # Compute gradients for interpretability
        class_idx = outputs.argmax(dim=1)
        target_score = outputs[0, class_idx]
        if gradient_required:
            target_score.backward()

        return outputs, class_idx, target_score

