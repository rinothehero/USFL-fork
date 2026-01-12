import copy
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from modules.trainer.model_trainer.propagator.base_propagator import BasePropagator
from transformers.models.distilbert.modeling_distilbert import (
    FFN,
    MultiHeadSelfAttention,
)

if TYPE_CHECKING:
    from client_args import Config
    from torch.nn import Module


class DistilbertPropagator(BasePropagator):
    def __init__(self, model: torch.nn.ModuleDict, config: "Config"):
        super().__init__()
        self.model = model
        self.outputs: torch.Tensor = None
        self.config = config

        key_with_position = next(
            filter(lambda key: "position_embeddings" in key, self.model.keys()), None
        )

        if key_with_position is not None:
            num_embeddings = self.model[key_with_position].num_embeddings
            self.register_buffer(
                "position_ids",
                torch.arange(num_embeddings).expand((1, -1)),
                persistent=False,
            )

        self.forward_mapper = {
            "word_embeddings": self.word_embeddings_forward,
            "position_embeddings": self.position_embeddings_forward,
            "attention": self.attention_forward,
            "ffn": self.ffn_forward,
            "pre_classifier": self.pre_classifier_forward,
        }

    def forward(
        self,
        input: Union[torch.Tensor, Tuple],
        params: dict = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        for layer_name, layer in self.model.items():
            layer_name = layer_name.split("-")[-1]
            mapped_forward = self.forward_mapper.get(layer_name)

            if "attention" in layer_name:
                input = (input, params["attention_mask"])

            module_output = (
                layer(input)
                if (mapped_forward is None)
                else mapped_forward(layer, input)
            )
            input = module_output

        self.outputs = module_output

        if isinstance(self.outputs, Tuple):
            outputs = list(self.outputs)
            outputs[0] = outputs[0].clone().detach().requires_grad_(True)
            outputs = tuple(outputs)
        else:
            outputs = self.outputs.clone().detach().requires_grad_(True)

        return outputs

    # Except for client, which contains the last part of the model.
    def backward(self, grads: torch.Tensor):
        if isinstance(self.outputs, Tuple):
            self.outputs = self.outputs[0]

        grads = grads.to(self.config.device)
        self.outputs.requires_grad_(True)
        self.outputs.backward(grads)

    def word_embeddings_forward(
        self,
        layer: torch.nn.Embedding,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if input_ids is not None:
            input_embeds = layer(input_ids)

        seq_length = input_embeds.size(1)

        return input_embeds, input_ids, seq_length

    def position_embeddings_forward(
        self,
        layer: torch.nn.Embedding,
        input_info: Tuple[torch.Tensor, torch.Tensor, int],
    ) -> torch.Tensor:

        input_embeds, input_ids, seq_length = input_info

        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length].to(input_embeds.device)
        else:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            ).to(input_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = layer(position_ids)
        embeddings = input_embeds + position_embeddings

        return embeddings

    def attention_forward(
        self,
        layer: MultiHeadSelfAttention,
        input_info: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        x, attn_mask = input_info
        head_mask = None  # Not needed in the current task.

        sa_output = layer(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
        )

        if type(sa_output) is not tuple:
            raise TypeError(
                f"sa_output must be a tuple but it is {type(sa_output)} type"
            )
        sa_output = sa_output[0]

        output = sa_output + x

        return output

    def ffn_forward(self, layer: FFN, sa_output: torch.Tensor) -> torch.Tensor:
        ffn_output = layer(sa_output)
        output = ffn_output + sa_output

        return output

    def pre_classifier_forward(
        self, layer: torch.nn.Linear, input: torch.Tensor
    ) -> torch.Tensor:
        input = input[:, 0]
        output = layer(input)
        output = torch.nn.ReLU()(output)

        return output
