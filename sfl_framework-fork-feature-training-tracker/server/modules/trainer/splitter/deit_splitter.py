"""
DeiT Splitter — delegates to DeiT.get_split_models() (Pattern B).

Same pattern as FlexibleResnetSplitter. Stage organizers detect
get_split_models() via hasattr() and bypass the splitter, but this
is registered for consistency.
"""

from typing import TYPE_CHECKING, List

from torch import nn

from .base_splitter import BaseSplitter

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module


class DeiTSplitter(BaseSplitter):
    def __init__(self, config: "Config"):
        self.config = config

    def split(self, model: "Module", params: dict) -> List[nn.Module]:
        if hasattr(model, "get_split_models"):
            client_model, server_model = model.get_split_models()
            return [client_model, server_model]
        raise ValueError(
            f"Model {type(model).__name__} does not support DeiT split. "
            "Expected model with get_split_models() method."
        )
