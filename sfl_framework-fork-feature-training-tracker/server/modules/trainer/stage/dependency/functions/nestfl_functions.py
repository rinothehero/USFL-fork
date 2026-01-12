from typing import TYPE_CHECKING

import torch

from ..torch_pruning import function

if TYPE_CHECKING:
    from server_args import Config


def restore_group(group, cfg: "Config"):
    for dep, idxs in group:
        layer = dep.layer
        prune_fn = dep.pruning_fn
        w = None
        b = None
        dim = 0

        if prune_fn in [
            function.prune_conv_out_channels,
            function.prune_linear_out_channels,
        ]:
            w = layer.weight.data

            if layer.bias is not None:
                b = layer.bias.data

        elif prune_fn in [
            function.prune_conv_in_channels,
            function.prune_linear_in_channels,
        ]:
            w = layer.weight.data

            if layer.bias is not None:
                b = layer.bias.data

        elif prune_fn == function.prune_batchnorm_out_channels:
            if layer.affine:
                w = layer.weight.data

            if layer.bias is not None:
                b = layer.bias.data

        elif prune_fn == function.prune_layernorm_out_channels:
            if layer.elementwise_affine:
                w = layer.weight.data

            if layer.bias is not None:
                b = layer.bias.data

        if w is not None:
            original_shape = list(w.shape)

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if prune_fn == function.prune_conv_out_channels:
                    layer.out_channels += len(idxs)
                elif prune_fn == function.prune_linear_out_channels:
                    layer.out_features += len(idxs)

                original_shape[0] += len(idxs)

            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if prune_fn == function.prune_conv_in_channels:
                    layer.in_channels += len(idxs)
                elif prune_fn == function.prune_linear_in_channels:
                    layer.in_features += len(idxs)

                dim = 1
                original_shape[1] += len(idxs)

            elif prune_fn in [
                function.prune_batchnorm_out_channels,
                function.prune_batchnorm_in_channels,
            ]:
                layer.num_features += len(idxs)
                non_pruned_running_idxs = [
                    i for i in range(layer.num_features) if i not in idxs
                ]

                restored_running_mean = torch.zeros(layer.num_features).to(cfg.device)
                restored_running_var = torch.zeros(layer.num_features).to(cfg.device)

                restored_running_mean[non_pruned_running_idxs] = layer.running_mean
                restored_running_var[non_pruned_running_idxs] = layer.running_var

                # restored_running_mean.index_fill_(0, torch.tensor(non_pruned_running_idxs).to(cfg.device), layer.running_mean)
                # restored_running_var.index_fill_(0, torch.tensor(non_pruned_running_idxs).to(cfg.device), layer.running_var)

                layer.running_mean = restored_running_mean
                layer.running_var = restored_running_var
                original_shape[0] += len(idxs)

            elif prune_fn in [
                function.prune_layernorm_out_channels,
                function.prune_layernorm_in_channels,
            ]:
                layer.normalized_shape += len(idxs)
                original_shape[0] += len(idxs)

            non_pruned_idxs = [i for i in range(original_shape[dim]) if i not in idxs]
            restored_matrix = torch.zeros(original_shape).to(cfg.device)
            restored_matrix = restored_matrix.index_add(
                dim, torch.tensor(non_pruned_idxs).to(cfg.device), w
            )

            pruned_idxs = torch.tensor(idxs, dtype=torch.int64).to(cfg.device)
            restored_matrix.index_fill_(dim, pruned_idxs, float("0"))

            layer.weight.data = restored_matrix

        if b is not None and prune_fn not in [
            function.prune_conv_in_channels,
            function.prune_linear_in_channels,
        ]:
            original_shape = list(b.shape)
            original_shape[0] += len(idxs)

            non_pruned_idxs = [i for i in range(original_shape[0]) if i not in idxs]
            restored_matrix = torch.zeros(original_shape).to(cfg.device)
            restored_matrix = restored_matrix.index_add(
                0, torch.tensor(non_pruned_idxs).to(cfg.device), b
            )

            pruned_idxs = torch.tensor(idxs, dtype=torch.int64).to(cfg.device)
            restored_matrix.index_fill_(0, pruned_idxs, float("0"))

            layer.bias.data = restored_matrix
