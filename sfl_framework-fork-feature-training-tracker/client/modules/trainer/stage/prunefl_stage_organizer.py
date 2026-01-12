from typing import TYPE_CHECKING

from modules.trainer.model_trainer.model_trainer import get_model_trainer

from .fl_stage_organizer import FLStageOrganizer
from .in_round.in_round import InRound
from .post_round.post_round import PostRound
from .pre_round.pre_round import PreRound

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.dataset.base_dataset import BaseDataset
    from modules.trainer.apis.common import CommonAPI


class PruneFLStageOrganizer(FLStageOrganizer):
    def __init__(
        self,
        config: "Config",
        server_config: "ServerConfig",
        api: "CommonAPI",
        dataset: "BaseDataset",
    ):
        super().__init__(config, server_config, api, dataset)
