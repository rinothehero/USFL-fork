import copy

import torch
import torch.nn as nn

from ...torch_pruning import pruner as tp
from ...torch_pruning.pruner import function


class EMA:
    def __init__(
        self,
        initial_model: nn.Module,
        ema_length: int,
        example_inputs: torch.Tensor,
        ignored_layers: list = [],
        root_module_types: list = [nn.Conv2d, nn.Linear],
        target_types: list = [
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
            nn.LayerNorm,
        ],
    ):
        self.ema_length = ema_length
        self.example_inputs = example_inputs
        self.ignored_layers = ignored_layers
        self.root_module_types = root_module_types
        self.prev_model = copy.deepcopy(initial_model)
        self.prev_model_ignored_layers = []

        prev_layers_dict = {
            name: module for name, module in self.prev_model.named_children()
        }
        for name, module in initial_model.named_children():
            if module in ignored_layers:
                self.prev_model_ignored_layers.append(prev_layers_dict[name])

        self.ema = None
        self.groups = None
        self.target_types = target_types

    def _calc_magnitude(self, group, ch_groups: int = 1):
        group_magnitude = []
        group_idxs = []

        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)

                local_magnitude = w.abs().sum(1)
                group_magnitude.append(local_magnitude)
                group_idxs.append(root_idxs)

                # if self.bias and layer.bias is not None:
                #     local_magnitude = layer.bias.data[idxs].abs()
                #     group_magnitude.append(local_magnitude)
                #     group_idxs.append(root_idxs)

            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)

                local_magnitude = w.abs().sum(1)

                if (
                    prune_fn == function.prune_conv_in_channels
                    and layer.groups != layer.in_channels
                    and layer.groups != 1
                ):
                    local_magnitude = local_magnitude.repeat(ch_groups)

                local_magnitude = local_magnitude[idxs]
                group_magnitude.append(local_magnitude)
                group_idxs.append(root_idxs)

            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_magnitude = w.abs()
                    group_magnitude.append(local_magnitude)
                    group_idxs.append(root_idxs)

                    # if self.bias and layer.bias is not None:
                    #     local_magnitude = layer.bias.data[idxs].abs()
                    #     group_magnitude.append(local_magnitude)
                    #     group_idxs.append(root_idxs)

            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    local_magnitude = w.abs()
                    group_magnitude.append(local_magnitude)
                    group_idxs.append(root_idxs)

                    # if self.bias and layer.bias is not None:
                    #     local_magnitude = layer.bias.data[idxs].abs()
                    #     group_magnitude.append(local_magnitude)
                    #     group_idxs.append(root_idxs)

        if len(group_magnitude) == 0:
            print("no mag")
            return None

        return group_magnitude

    def update(self, model: nn.Module):
        group_emas = []
        groups = []

        current_DG = tp.DependencyGraph().build_dependency(
            model, example_inputs=self.example_inputs
        )
        current_groups = current_DG.get_all_groups(ignored_layers=self.ignored_layers)

        prev_DG = tp.DependencyGraph().build_dependency(
            self.prev_model, example_inputs=self.example_inputs
        )
        prev_groups = prev_DG.get_all_groups(
            ignored_layers=self.prev_model_ignored_layers
        )

        for i, (current_group, prev_group) in enumerate(
            zip(current_groups, prev_groups)
        ):
            current_group_magnitude = self._calc_magnitude(current_group)
            prev_group_magnitude = self._calc_magnitude(prev_group)

            diff = list(
                map(
                    lambda x, y: (x - y).abs(),
                    current_group_magnitude,
                    prev_group_magnitude,
                )
            )
            diff_sum = sum(diff)

            groups.append(current_group[0][0].target._name)
            group_emas.append(diff_sum)

        if self.ema is None:
            self.ema = group_emas
        else:
            alpha = 2 / (self.ema_length + 1)
            self.ema = list(
                map(lambda x, y: x * alpha + y * (1 - alpha), group_emas, self.ema)
            )

        print("prev groups")
        print(self.groups)
        print("groups")
        print(groups)
        self.groups = groups
        self.prev_model = copy.deepcopy(model)

        prev_model_ignored_layers = []
        prev_layers_dict = {
            name: module for name, module in self.prev_model.named_children()
        }
        for name, module in model.named_children():
            if module in self.ignored_layers:
                prev_model_ignored_layers.append(prev_layers_dict[name])
        self.prev_model_ignored_layers = prev_model_ignored_layers

    def get_ema_threshold(self, group: int, percentile: float):
        # ema_1d = []
        # for ema in self.ema:
        #     ema_1d.extend(ema.tolist())

        # ema_1d = torch.tensor(ema_1d)
        # ema_1d = ema_1d[ema_1d != 0]
        # # print(f'ema_1d: {ema_1d}')
        # kth_value = torch.kthvalue(ema_1d, int(len(ema_1d) * percentile))
        # # print(kth_value)
        try:
            group_idx = self.groups.index(group[0][0].target._name)
            # print(len(self.ema))
            # print(group_idx)
            target_ema = self.ema[group_idx].tolist()
            # print(target_ema)
            target_tensor = torch.tensor(target_ema)
            target_tensor = target_tensor[target_tensor != 0]
            target_kth_value = torch.kthvalue(
                target_tensor, int(len(target_tensor) * percentile)
            )
            return target_kth_value[0]
        except Exception as e:
            print(e)
            return torch.inf
        # return ema_1d[kth_value[1]].item()

    def get_ema_mask(self, group, group_incidies, threshold):
        try:
            group_idx = self.groups.index(group[0][0].target._name)
            ema = self.ema[group_idx]
            mask = torch.zeros_like(ema)
            mask[ema <= threshold] = 1

            return mask[group_incidies]
        except:
            return torch.zeros_like(torch.tensor(group_incidies))
