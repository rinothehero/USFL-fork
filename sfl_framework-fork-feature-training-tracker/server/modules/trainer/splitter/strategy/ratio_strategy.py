from collections import deque
from typing import TYPE_CHECKING

from .base_strategy import BaseStrategy
from utils.log_utils import vprint

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module


class RatioStrategy(BaseStrategy):
    def __init__(self, config: "Config"):
        self.config = config

    def _ratio_param(self, model: "Module", params: dict):
        not_splitable_modules = tuple(params.get("not_splitable_modules", []))
        split_ratios = deque(sorted(params["split_ratio"]))
        total_params = sum(p.numel() for p in model.parameters())
        split_points = []
        accumulated_params = 0

        pending_split_point = None
        pending_split_ratio = None

        def recurse_modules(parent, prefix=""):
            nonlocal accumulated_params, pending_split_point, pending_split_ratio

            for name, child in parent.named_children():
                if isinstance(child, not_splitable_modules):
                    vprint(f"{prefix}{name} is not splitable ({type(child)})", 2)

                if len(list(child.named_children())) != 0 and not isinstance(
                    child, not_splitable_modules
                ):
                    recurse_modules(child, prefix + name + ".")
                else:
                    layer_params = sum(p.numel() for p in child.parameters())

                    accumulated_params += layer_params
                    current_ratio = accumulated_params / total_params
                    vprint(
                        f"{prefix}{name} has {layer_params} params, current_ratio {current_ratio}", 2
                    )

                    if pending_split_point is not None:
                        if current_ratio > pending_split_ratio:
                            vprint(f"layer {pending_split_point} is split point", 2)
                            vprint(
                                f"current_ratio {current_ratio} > pending_split_ratio {pending_split_ratio}", 2
                            )
                            split_points.append(pending_split_point)
                            pending_split_point = None
                            pending_split_ratio = None

                            if not split_ratios:
                                break

                        else:
                            pending_split_point = prefix + name
                            continue

                    if split_ratios and current_ratio >= split_ratios[0]:
                        vprint(f"layer {prefix + name} is pending split point", 2)
                        vprint(f"pending_split_ratio {current_ratio}", 2)
                        pending_split_ratio = current_ratio
                        pending_split_point = prefix + name
                        split_ratios.popleft()

        recurse_modules(model)

        if pending_split_point is not None:
            split_points.append(pending_split_point)

        return split_points

    def _ratio_layer(self, model: "Module", params: dict):
        not_splitable_modules = tuple(params.get("not_splitable_modules", []))
        split_ratios = deque(sorted(params["split_ratio"]))

        def __count_total_layers(module):
            total = 0
            for child in module.children():
                if isinstance(child, not_splitable_modules):
                    total += 1
                elif len(list(child.children())) > 0:
                    total += __count_total_layers(child)
                else:
                    total += 1
            return total

        total_layers = __count_total_layers(model)
        split_points = []
        accumulated_layers = 0

        def recurse_layers(parent, prefix=""):
            nonlocal accumulated_layers

            for name, child in parent.named_children():
                full_name = prefix + name

                if isinstance(child, not_splitable_modules):
                    accumulated_layers += 1
                    current_ratio = accumulated_layers / total_layers
                    vprint(
                        f"{full_name} (not_splitable: {type(child)}) is treated as a single layer {accumulated_layers}/{total_layers}, current_ratio {current_ratio}", 2
                    )

                    while split_ratios and current_ratio >= split_ratios[0]:
                        vprint(f"layer {full_name} is split point", 2)
                        split_points.append(full_name)
                        split_ratios.popleft()

                elif len(list(child.children())) > 0:
                    recurse_layers(child, full_name + ".")
                else:
                    accumulated_layers += 1
                    current_ratio = accumulated_layers / total_layers
                    vprint(
                        f"{full_name} is layer {accumulated_layers}/{total_layers}, current_ratio {current_ratio}", 2
                    )

                    while split_ratios and current_ratio >= split_ratios[0]:
                        vprint(f"layer {full_name} is split point", 2)
                        split_points.append(full_name)
                        split_ratios.popleft()

        recurse_layers(model)

        return split_points

    def find_split_points(self, model: "Module", params: dict):
        if "split_ratio" not in params:
            raise ValueError("Split ratio is not defined")

        if self.config.split_strategy == "ratio_param":
            return self._ratio_param(model, params)
        elif self.config.split_strategy == "ratio_layer":
            return self._ratio_layer(model, params)
        else:
            raise ValueError(f"Invalid split strategy: {self.config.split_strategy}")
