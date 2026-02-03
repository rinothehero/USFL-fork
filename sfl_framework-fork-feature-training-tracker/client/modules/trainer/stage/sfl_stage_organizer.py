import asyncio
import time
from typing import TYPE_CHECKING

from modules.trainer.model_trainer.model_trainer import get_model_trainer

from .base_stage_organizer import BaseStageOrganizer
from .in_round.in_round import InRound
from .post_round.post_round import PostRound
from .pre_round.pre_round import PreRound

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.dataset.base_dataset import BaseDataset
    from modules.trainer.apis.common import CommonAPI


class SFLStageOrganizer(BaseStageOrganizer):
    def __init__(
        self,
        config: "Config",
        server_config: "ServerConfig",
        api: "CommonAPI",
        dataset: "BaseDataset",
    ):
        self.config = config
        self.server_config = server_config
        self.api = api
        self.dataset = dataset
        self.trainloader = self.dataset.get_trainloader()

        self.pre_round = PreRound(config, server_config)
        self.in_round = InRound(config, server_config)
        self.post_round = PostRound(config, server_config)

        self.model = None
        self.training_params = None
        self.model_trainer = None

    async def _pre_round(self):
        await self.pre_round.notify_client_information(
            self.api, filter=["dataset"], dataset=self.dataset
        )
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

    async def _in_round(self):
        await self.in_round.train(self.model_trainer)

    async def _post_round(self):
        import pickle

        submit_params = {
            "dataset_size": len(self.dataset.get_trainset()),
        }

        # G Measurement: Include client gradient if collected
        if (
            hasattr(self.model_trainer, "measurement_gradient")
            and self.model_trainer.measurement_gradient is not None
        ):
            submit_params["client_gradient"] = pickle.dumps(
                self.model_trainer.measurement_gradient
            ).hex()
            if self.model_trainer.measurement_gradient_weight is not None:
                submit_params["measurement_gradient_weight"] = (
                    self.model_trainer.measurement_gradient_weight
                )
            self.model_trainer.measurement_gradient = None
            self.model_trainer.measurement_gradient_weight = None
        elif (
            hasattr(self.model_trainer, "accumulated_gradients")
            and self.model_trainer.accumulated_gradients
        ):
            grad_data = self.model_trainer.accumulated_gradients[0]
            submit_params["client_gradient"] = pickle.dumps(grad_data).hex()
            if self.model_trainer.gradient_weights:
                submit_params["measurement_gradient_weight"] = (
                    self.model_trainer.gradient_weights[0]
                )
            self.model_trainer.accumulated_gradients.clear()
            self.model_trainer.gradient_weights.clear()

        # Drift Measurement: Include drift metrics if collected
        if (
            hasattr(self.model_trainer, "enable_drift_measurement")
            and self.model_trainer.enable_drift_measurement
        ):
            drift_metrics = self.model_trainer.get_drift_metrics()
            submit_params["drift_trajectory_sum"] = drift_metrics["drift_trajectory_sum"]
            submit_params["drift_batch_steps"] = drift_metrics["drift_batch_steps"]
            submit_params["drift_endpoint"] = drift_metrics["drift_endpoint"]

        await self.post_round.submit_model(
            self.api,
            self.model,
            self.training_params["signiture"],
            submit_params,
        )

    async def run_pre_round(self):
        return await self._pre_round()

    async def run_in_round(self):
        task = asyncio.create_task(self._in_round())

        while not task.done():
            if time.time() > self.training_params["round_end_time"]:
                print("Round end time reached, cancelling in round")
                task.cancel()
                return

            await asyncio.sleep(0.0001)

    async def run_post_round(self):
        task = asyncio.create_task(self._post_round())

        while not task.done():
            if time.time() > self.training_params["round_end_time"]:
                print("Round end time reached, cancelling post round")
                task.cancel()
                return

            await asyncio.sleep(0.0001)
