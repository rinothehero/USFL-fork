from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class MultiSFLConfig:
    num_rounds: int = 50
    num_clients_total: int = 100
    n_main_clients_per_round: int = 10
    num_branches: Optional[int] = None
    dataset: str = "cifar10"
    num_classes: int = 10
    device: str = "cpu"
    seed: int = 42

    batch_size: int = 64
    min_samples_per_client: int = 10
    use_full_epochs: bool = False
    local_steps: int = 5
    lr_client: float = 0.01
    lr_server: float = 0.01
    momentum: float = 0.9

    alpha_master_pull: float = 0.1

    gamma: float = 0.5

    p0: float = 0.05
    p_min: float = 0.01
    p_max: float = 0.5
    eps: float = 1e-12

    max_assistant_trials_per_branch: int = 20
    replay_budget_mode: Literal["batch", "local_dataset"] = "local_dataset"
    replay_min_total: int = 0

    p_update: Literal["paper", "abs_ratio", "one_plus_delta"] = "abs_ratio"
    delta_clip: float = 0.2

    # Model Architecture
    model_type: Literal[
        "simple",
        "alexnet",
        "alexnet_light",
        "resnet18",
        "resnet18_light",
        "resnet18_flex",
        "resnet18_image_style",
    ] = "simple"
    split_layer: Optional[str] = None

    # G Measurement
    enable_g_measurement: bool = True
    g_measure_frequency: int = 10
    use_variance_g: bool = False
    use_sfl_transform: bool = False
    oracle_mode: Literal["master", "branch"] = "master"

    clip_grad: bool = False
    clip_grad_max_norm: float = 10.0

    def __post_init__(self):
        if self.p_min <= 0:
            raise ValueError(
                f"p_min must be > 0 to prevent replay dying, got {self.p_min}"
            )
        if self.num_branches is None:
            self.num_branches = self.n_main_clients_per_round
        if self.split_layer is None and self.model_type in ["alexnet", "alexnet_light"]:
            is_grayscale = self.dataset in ["mnist", "fmnist"]
            if self.model_type == "alexnet_light" and not is_grayscale:
                self.split_layer = "conv.2"
            else:
                self.split_layer = "conv.5"
