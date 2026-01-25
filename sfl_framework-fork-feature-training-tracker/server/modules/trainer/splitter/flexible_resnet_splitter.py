"""
FlexibleResnetSplitter - Layer boundary based split for ResNet.

Unlike ResnetSplitter which uses leaf module names and ModuleDict,
this splitter returns pre-built FlexibleResNetClient and FlexibleResNetServer models
that support layer boundary splits (layer2, layer3, layer4).
"""

from typing import TYPE_CHECKING, List
from torch import nn

from .base_splitter import BaseSplitter

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module


class FlexibleResnetSplitter(BaseSplitter):
    """
    Splitter for FlexibleResNet models.

    This splitter doesn't actually split the model - it returns the pre-built
    client and server models from FlexibleResNet.get_split_models().

    The FlexibleResNet model handles all the complexity of:
    - Layer boundary splits (layer2, layer3, layer4) -> Tensor output
    - Mid-block splits (layer1.1.bn2) -> Tuple output with residual handling
    """

    def __init__(self, config: "Config"):
        self.config = config

    def split(self, model: "Module", params: dict) -> List[nn.Module]:
        """
        Return pre-built split models from FlexibleResNet.

        Args:
            model: FlexibleResNet instance
            params: Additional parameters (unused for flexible split)

        Returns:
            List of [client_model, server_model]
        """
        # FlexibleResNet provides get_split_models() method
        if hasattr(model, "get_split_models"):
            client_model, server_model = model.get_split_models()
            return [client_model, server_model]

        raise ValueError(
            f"Model {type(model).__name__} does not support flexible split. "
            "Use model='resnet18_flex' with FlexibleResNet."
        )
