"""
DeiT-S (Data-efficient Image Transformer - Small) for SFL Framework.

Split at transformer block boundaries using 'blocks.N' format:
  Client: patch_embed + pos_embed + cls_token + blocks[0:N]
  Server: blocks[N:12] + norm + head

Activation at split point: plain Tensor (batch, num_patches+1, embed_dim).

Usage:
    config.model = "deit_s"
    config.split_layer = "blocks.3"   # client=3 blocks, server=9 blocks

Requires: timm >= 0.9.0
"""

from typing import TYPE_CHECKING, Tuple

import torch
from torch import nn

from .base_model import BaseModel

if TYPE_CHECKING:
    from server_args import Config
    from torch.utils.data import DataLoader


NUM_BLOCKS = 12


def _parse_deit_split_layer(split_layer: str) -> int:
    """Parse 'blocks.N' → N (number of blocks on client side).

    Examples:
        "blocks.3"  → 3  (client gets blocks 0-2, server gets blocks 3-11)
        "blocks.0"  → 0  (client gets patch_embed only)
        "blocks.11" → 11 (client gets blocks 0-10, server gets block 11)
    """
    if not split_layer.startswith("blocks."):
        raise ValueError(
            f"DeiT split_layer must be 'blocks.N' format (N=0..{NUM_BLOCKS}), "
            f"got '{split_layer}'"
        )
    n = int(split_layer.split(".")[1])
    if n < 0 or n > NUM_BLOCKS:
        raise ValueError(
            f"DeiT split_layer block index must be 0..{NUM_BLOCKS}, got {n}"
        )
    return n


def _create_deit_base(num_classes: int, img_size: int, patch_size: int):
    """Create base DeiT-S model via timm."""
    import timm

    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=False,
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
    )
    return model


class DeiTClient(nn.Module):
    """Client-side DeiT: patch_embed + cls_token + pos_embed + first N blocks."""

    def __init__(self, patch_embed, cls_token, pos_embed, blocks, pos_drop):
        super().__init__()
        self.patch_embed = patch_embed
        self.cls_token = cls_token  # nn.Parameter (1, 1, 384)
        self.pos_embed = pos_embed  # nn.Parameter (1, num_patches+1, 384)
        self.blocks = blocks  # nn.Sequential of first N blocks
        self.pos_drop = pos_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        return x  # (B, num_patches+1, embed_dim)


class DeiTServer(nn.Module):
    """Server-side DeiT: remaining blocks + LayerNorm + classification head."""

    def __init__(self, blocks, norm, head):
        super().__init__()
        self.blocks = blocks  # nn.Sequential of remaining blocks
        self.norm = norm  # LayerNorm
        self.head = head  # Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]  # CLS token only
        x = self.head(x)
        return x


class DeiT(BaseModel):
    """DeiT-S wrapper for SFL Framework following FlexibleResNet pattern."""

    def __init__(self, config: "Config", num_classes: int):
        super().__init__(config)
        self.config = config
        self.num_classes = num_classes

        split_layer = getattr(config, "split_layer", "blocks.3")
        self.split_layer = split_layer
        self.split_n = _parse_deit_split_layer(split_layer)

        # Determine image/patch size from dataset
        if config.dataset in ("cifar10", "cifar100", "fmnist", "mnist"):
            img_size, patch_size = 32, 4
        else:
            img_size, patch_size = 224, 16

        # Build full model from timm
        full = _create_deit_base(num_classes, img_size, patch_size)

        # Verify expected attributes
        for attr in ("patch_embed", "cls_token", "pos_embed", "blocks", "norm", "head"):
            if not hasattr(full, attr):
                raise AttributeError(
                    f"timm DeiT model missing expected attribute '{attr}'. "
                    f"Check timm version (need >= 0.9.0)."
                )

        # Split blocks
        all_blocks = list(full.blocks.children())
        client_blocks = nn.Sequential(*all_blocks[: self.split_n])
        server_blocks = nn.Sequential(*all_blocks[self.split_n :])

        self.client_model = DeiTClient(
            patch_embed=full.patch_embed,
            cls_token=full.cls_token,
            pos_embed=full.pos_embed,
            blocks=client_blocks,
            pos_drop=full.pos_drop if hasattr(full, "pos_drop") else nn.Identity(),
        )
        self.server_model = DeiTServer(
            blocks=server_blocks,
            norm=full.norm,
            head=full.head,
        )

        # Full model for evaluation (independent copy)
        self.torch_model = _create_deit_base(num_classes, img_size, patch_size)
        self.sync_full_model_from_split()

    def get_split_models(self) -> Tuple[nn.Module, nn.Module]:
        """Return (client_model, server_model) tuple."""
        return self.client_model, self.server_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass (for evaluation)."""
        return self.torch_model(x)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.torch_model.eval()
        with torch.no_grad():
            return self.forward(inputs)

    def sync_full_model_from_split(self):
        """Sync full model parameters from split client/server models.

        Client keys map directly to full model (same block indices).
        Server block keys need re-indexing: server blocks.0 → full blocks.{split_n}.
        """
        full_state = self.torch_model.state_dict()

        # Client: keys like patch_embed.*, cls_token, pos_embed, blocks.0.*, blocks.1.*
        # These map directly (client block indices match full model)
        for key, value in self.client_model.state_dict().items():
            if key in full_state:
                full_state[key] = value.clone()

        # Server: blocks need re-indexing, norm.* and head.* map directly
        for key, value in self.server_model.state_dict().items():
            if key.startswith("blocks."):
                # "blocks.0.attn.qkv.weight" → "blocks.{0+split_n}.attn.qkv.weight"
                dot_pos = key.index(".", len("blocks."))
                local_idx = int(key[len("blocks.") : dot_pos])
                rest = key[dot_pos:]
                full_key = f"blocks.{local_idx + self.split_n}{rest}"
            else:
                full_key = key

            if full_key in full_state:
                full_state[full_key] = value.clone()

        self.torch_model.load_state_dict(full_state)

    def evaluate(self, testloader: "DataLoader") -> float:
        """Evaluate using full model (synced from split)."""
        self.sync_full_model_from_split()
        self.torch_model.eval()
        self.torch_model.to(self.config.device)
        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def get_torch_model(self) -> nn.Module:
        return self.torch_model

    def set_torch_model(self, torch_model: nn.Module):
        torch_model.to(self.config.device)
        self.torch_model = torch_model

    def save_model(self, save_path: str) -> None:
        torch.save(
            {
                "client": self.client_model.state_dict(),
                "server": self.server_model.state_dict(),
                "full": self.torch_model.state_dict(),
            },
            save_path + ".pth",
        )

    def load_model(self, load_path: str) -> None:
        state = torch.load(load_path + ".pth")
        self.client_model.load_state_dict(state["client"])
        self.server_model.load_state_dict(state["server"])
        self.torch_model.load_state_dict(state["full"])
