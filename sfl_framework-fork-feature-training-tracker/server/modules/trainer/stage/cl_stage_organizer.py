from typing import TYPE_CHECKING

from .base_stage_organizer import BaseStageOrganizer
from .dependency.model_trainer.model_trainer import get_model_trainer
from .in_round.in_round import InRound
from .post_round.post_round import PostRound
from .pre_round.pre_round import PreRound

if TYPE_CHECKING:
    from server_args import Config

    from ...dataset.base_dataset import BaseDataset
    from ...global_dict.global_dict import GlobalDict
    from ...model.base_model import BaseModel


class CLStageOrganizer(BaseStageOrganizer):
    def __init__(
        self,
        config: "Config",
        global_dict: "GlobalDict",
        model: "BaseModel",
        dataset: "BaseDataset",
    ):
        self.config = config
        self.global_dict = global_dict
        self.model = model

        self.torch_model = model.get_torch_model()
        self.dataset = dataset
        self.testloader = dataset.get_testloader()

        self.pre_round = PreRound(config, global_dict)
        self.in_round = InRound(config, global_dict)
        self.post_round = PostRound(config, global_dict)

        self.round_start_time = 0
        self.round_end_time = 0

        self.parameters = self._count_parameters(self.model.get_torch_model())

    async def _pre_round(self, round_number: int):
        self.round_start_time, self.round_end_time = (
            self.pre_round.calculate_round_end_time()
        )
        self.global_dict.add_event(
            "ROUND_START",
            {
                "strat_timestamp": self.round_start_time,
                "end_timestamp": self.round_end_time,
            },
        )

    async def _in_round(self, round_number: int):
        model_trainer = get_model_trainer(
            self.config, self.dataset, self.torch_model, {"local_epochs": 1}
        )
        await model_trainer.train()

    async def _post_round(self, round_number: int):
        accuracy = self.post_round.evaluate_global_model(self.model, self.testloader)
        self.global_dict.add_event("MODEL_EVALUATED", {"accuracy": accuracy})

        print(f"[Round {round_number}] Accuracy: {accuracy}")
