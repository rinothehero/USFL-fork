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


class NestFLStageOrganizer(FLStageOrganizer):
    def __init__(
        self,
        config: "Config",
        server_config: "ServerConfig",
        api: "CommonAPI",
        dataset: "BaseDataset",
    ):
        super().__init__(config, server_config, api, dataset)

    async def _pre_round(self):
        await self.pre_round.notify_client_information(self.api, filter=["cpu"])
        await self.pre_round.notify_wait_for_training(self.api)

        model, training_params = await self.pre_round.wait_for_start_round(self.api)
        
        if (model is None) and (training_params is None):
            return True
        
        self.model = model
        self.training_params = training_params

        self.model_trainer = get_model_trainer(
            self.config,
            self.server_config,
            self.dataset,
            self.model,
            self.training_params,
            self.api,
        )

        print("Initialized model trainer")

        return False