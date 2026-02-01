from typing import TYPE_CHECKING

from .fedsparsify_stage_organizer import FedSparsifyStageOrganizer
from .fitfl_stage_organizer import FitFLStageOrganizer
from .fl_stage_organizer import FLStageOrganizer
from .mix2sfl_stage_organizer import Mix2SFLStageOrganizer
from .nestfl_stage_organizer import NestFLStageOrganizer
from .prunefl_stage_organizer import PruneFLStageOrganizer
from .scala_stage_organizer import ScalaStageOrganizer
from .scaffold_sfl_stage_organizer import ScaffoldSFLStageOrganizer
from .sfl_stage_organizer import SFLStageOrganizer
from .usfl_stage_organizer import USFLStageOrganizer

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.dataset.base_dataset import BaseDataset
    from modules.trainer.apis.common import CommonAPI


def get_stage_organizer(
    config: "Config",
    server_config: "ServerConfig",
    api: "CommonAPI",
    dataset: "BaseDataset",
):
    if server_config.method in ["fl", "fedprox", "fedcbs"]:
        return FLStageOrganizer(config, server_config, api, dataset)
    elif server_config.method == "fitfl":
        return FitFLStageOrganizer(config, server_config, api, dataset)
    elif server_config.method == "prunefl":
        return PruneFLStageOrganizer(config, server_config, api, dataset)
    elif server_config.method == "fedsparsify":
        return FedSparsifyStageOrganizer(config, server_config, api, dataset)
    elif server_config.method == "nestfl":
        return NestFLStageOrganizer(config, server_config, api, dataset)
    elif server_config.method in ["sfl", "sflprox"]:
        return SFLStageOrganizer(config, server_config, api, dataset)
    elif server_config.method == "scaffold_sfl":
        return ScaffoldSFLStageOrganizer(config, server_config, api, dataset)
    elif server_config.method == "mix2sfl":
        return Mix2SFLStageOrganizer(config, server_config, api, dataset)
    elif server_config.method == "scala":
        return ScalaStageOrganizer(config, server_config, api, dataset)
    elif server_config.method == "usfl":
        return USFLStageOrganizer(config, server_config, api, dataset)
    else:
        raise ValueError(f"Invalid method: {server_config.method}")
