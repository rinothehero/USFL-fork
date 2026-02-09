from typing import TYPE_CHECKING

from utils.log_utils import vprint

from .base_stage_organizer import BaseStageOrganizer
from .decorator.disconnect_handler import disconnect_handler
from .in_round.in_round import InRound
from .post_round.post_round import PostRound
from .pre_round.pre_round import PreRound

if TYPE_CHECKING:
    from server_args import Config

    from ...dataset.base_dataset import BaseDataset
    from ...global_dict.global_dict import GlobalDict
    from ...model.base_model import BaseModel
    from ...trainer.aggregator.base_aggregator import BaseAggregator
    from ...trainer.seletor.base_selector import BaseSelector
    from ...ws.connection import Connection


class FLStageOrganizer(BaseStageOrganizer):
    def __init__(
        self,
        config: "Config",
        connection: "Connection",
        global_dict: "GlobalDict",
        aggregator: "BaseAggregator",
        model: "BaseModel",
        dataset: "BaseDataset",
        selector: "BaseSelector",
    ):
        self.config = config
        self.connection = connection
        self.global_dict = global_dict
        self.aggregator = aggregator
        self.model = model
        self.testloader = dataset.get_testloader()
        self.num_classes = dataset.get_num_classes()
        self.selector = selector

        self.pre_round = PreRound(config, global_dict)
        self.in_round = InRound(config, global_dict)
        self.post_round = PostRound(config, global_dict)

        self.selected_clients = []
        self.round_start_time = 0
        self.round_end_time = 0

        self.parameters = self._count_parameters(self.model.get_torch_model())

    @disconnect_handler()
    async def _pre_round(self, round_number: int):
        await self.pre_round.wait_for_client_informations()
        await self.pre_round.wait_for_clients()

        client_informations = self.global_dict.get("client_informations")
        self.selected_clients = self.pre_round.select_clients(
            self.selector,
            self.connection,
            {
                "client_informations": client_informations,
                "num_classes": self.num_classes,
                "batch_size": self.config.batch_size,
            },
        )

        self.global_dict.add_event(
            "CLIENTS_SELECTED", {"client_ids": self.selected_clients}
        )

        self.round_start_time, self.round_end_time = (
            self.pre_round.calculate_round_end_time()
        )
        model_queue = self.global_dict.get("model_queue")
        model_queue.start_insert_mode()

        self.global_dict.add_event(
            "ROUND_START",
            {
                "strat_timestamp": self.round_start_time,
                "end_timestamp": self.round_end_time,
            },
        )

        await self.pre_round.send_global_model(
            self.selected_clients,
            self.model,
            self.connection,
            {
                "round_number": round_number,
                "round_end_time": self.round_end_time,
                "round_start_time": self.round_start_time,
                "signiture": model_queue.get_signiture(),
                "local_epochs": self.config.local_epochs,
            },
        )

    @disconnect_handler()
    async def _in_round(self, round_number: int):
        await self.in_round.wait_for_model_submission(
            self.selected_clients, self.round_end_time
        )

    @disconnect_handler()
    async def _post_round(self, round_number: int):
        model_queue = self.global_dict.get("model_queue")
        model_queue.end_insert_mode()

        client_ids = [model[0] for model in model_queue.queue]
        self.global_dict.add_event(
            "MODEL_AGGREGATION_START", {"client_ids": client_ids}
        )
        updated_torch_model = self.post_round.aggregate_models(
            self.aggregator, model_queue
        )
        self.global_dict.add_event("MODEL_AGGREGATION_END", {"client_ids": client_ids})

        if updated_torch_model != None:
            self.post_round.update_global_model(updated_torch_model, self.model)
            vprint("Updated global model", 2)

        model_queue.clear()
        accuracy = self.post_round.evaluate_global_model(self.model, self.testloader)
        self.global_dict.add_event("MODEL_EVALUATED", {"accuracy": accuracy})

        vprint(f"[Round {round_number}] Accuracy: {accuracy}", 1)
