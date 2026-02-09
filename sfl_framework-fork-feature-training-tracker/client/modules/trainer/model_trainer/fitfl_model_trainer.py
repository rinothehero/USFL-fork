import asyncio
import math
import time
from typing import TYPE_CHECKING

from utils.log_utils import vprint

from .base_model_trainer import BaseModelTrainer

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.dataset.base_dataset import BaseDataset
    from modules.trainer.apis.common import CommonAPI
    from torch.nn import Module


class FitFLModelTrainer(BaseModelTrainer):
    def __init__(
        self,
        config: "Config",
        server_config: "ServerConfig",
        dataset: "BaseDataset",
        model: "Module",
        training_params: dict,
    ):
        self.config = config
        self.server_config = server_config
        self.dataset = dataset
        self.model = model
        self.training_params = training_params
        self.criterion = self.get_criterion(server_config)
        self.trainloader = self.dataset.get_trainloader()

        self.toggle = True
        self.parameters = self._count_parameters(model)
        self.prev_learnable_module_count = -1
        self.alpha_updated = 0
        self.cur_freezing_ratio = 0
        self.alpha = 0

    def _count_parameters(self, model):
        total_parameters = 0
        for param in model.parameters():
            total_parameters += param.nelement()
        return total_parameters

    def _freezing(self, ratio, prev_ratio):
        vprint(
            f"Client freezing the model (freeze {ratio * 100:.3f}% of the parameters)",
            2,
        )

        keep_ratio = 1 - ratio
        keep_parameters = self.parameters * keep_ratio
        cumulated_parameters = 0
        learnable_module_count = 0

        parameters = (
            list(self.model.parameters())
            if self.toggle
            else reversed(list(self.model.parameters()))
        )

        for params in parameters:
            module_parameters = params.nelement()
            cumulated_parameters += module_parameters

            if cumulated_parameters <= keep_parameters:
                # If current ratio is bigger than prev_ratio, freeze at least one more layer.
                if (
                    ratio > prev_ratio
                    and self.prev_learnable_module_count - 1 == learnable_module_count
                ):
                    params.requires_grad = False
                else:
                    params.requires_grad = True
                    learnable_module_count += 1
            else:
                # If all the modules are freezed, then set the last module as learnable
                if learnable_module_count == 0:
                    params.requires_grad = True
                    learnable_module_count += 1
                else:
                    params.requires_grad = False

        self.prev_learnable_module_count = learnable_module_count
        vprint(f"Client frozen the model (keep {learnable_module_count} modules)", 2)

    def _custom_sigmoid(self, x):
        return 1 / (1 + math.exp(4 * -x)) - 0.5

    def _update_alpha(self, epoch_elapsed_time, alpha_readjust_expected_time):
        diff_ratio = (
            epoch_elapsed_time - alpha_readjust_expected_time
        ) / alpha_readjust_expected_time

        new_alpha = self._custom_sigmoid(self.alpha + diff_ratio)
        self.alpha = new_alpha

        vprint(f"Client alpha: {self.alpha:.3f}", 2)

    async def _train_default_dataset(self, params: dict):
        self.model.train()
        trained_epochs = 0
        optimizer = self.get_optimizer(self.server_config)
        num_extra_epoch = 0

        local_epochs = self.training_params["local_epochs"]
        round_time = params["expected_round_end_time"] - time.time()
        alpha_readjust = False
        alpha_readjust_expected_time = 0
        self.toggle = not self.toggle
        training_time_per_epochs = []

        vprint(f"Client starts training for {local_epochs} epochs", 2)
        vprint(f"Client has {round_time:.3f}s for training", 2)

        self.cur_freezing_ratio = 0
        self._freezing(self.cur_freezing_ratio, self.cur_freezing_ratio)

        while True:
            running_loss = 0.0

            epoch_start_time = time.time()
            trained_epochs += 1

            for i, batch in enumerate(self.trainloader):
                x, y = batch
                x, y = x.to(self.config.device), y.to(self.config.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                await asyncio.to_thread(optimizer.zero_grad)
                await asyncio.to_thread(loss.backward)
                await asyncio.to_thread(optimizer.step)

                running_loss += loss.item()

            epoch_end_time = time.time()
            epoch_elapsed_time = epoch_end_time - epoch_start_time
            training_time_per_epochs.append(epoch_elapsed_time)
            round_time = round_time - epoch_elapsed_time

            vprint(f"[Epoch {trained_epochs}/{local_epochs}] loss: {loss:.3f}", 2)

            if alpha_readjust:
                self._update_alpha(epoch_elapsed_time, alpha_readjust_expected_time)

            if round_time < 0:
                break

            if trained_epochs < local_epochs:
                left_epochs = local_epochs - trained_epochs
                desired_time_per_epoch = round_time / left_epochs
                estimated_time = epoch_elapsed_time * left_epochs

                vprint(
                    f"Estimated time left: {estimated_time:.3f}s | Desired epoch time: {desired_time_per_epoch:.3f}s",
                    2,
                )
                vprint(
                    f"Round time: {round_time:.3f}s | Epoch time: {epoch_elapsed_time:.3f}s",
                    2,
                )

                if round_time < training_time_per_epochs[-1]:
                    break

                if desired_time_per_epoch < epoch_elapsed_time:
                    alpha_readjust = True

                    overrun_ratio = 1 - (desired_time_per_epoch / epoch_elapsed_time)
                    unfrozen_ratio = 1 - self.cur_freezing_ratio

                    additional_freezing_ratio = unfrozen_ratio * overrun_ratio
                    freezing_ratio = (
                        self.cur_freezing_ratio + additional_freezing_ratio + self.alpha
                    )

                    if freezing_ratio >= 1:
                        alpha_readjust = False
                        freezing_ratio = self.cur_freezing_ratio

                    if freezing_ratio < 0:
                        freezing_ratio = 0

                    prev_ratio = self.cur_freezing_ratio
                    self.cur_freezing_ratio = freezing_ratio
                    alpha_readjust_expected_time = desired_time_per_epoch

                    vprint(f"Client freezing ratio: {freezing_ratio:.3f}", 2)

                    self._freezing(freezing_ratio, prev_ratio)

                else:
                    if self.cur_freezing_ratio > 0:
                        alpha_readjust = True

                        underrun_ratio = (
                            desired_time_per_epoch / epoch_elapsed_time
                        ) - 1
                        frozen_ratio = self.cur_freezing_ratio

                        additional_freezing_ratio = frozen_ratio * underrun_ratio
                        freezing_ratio = (
                            self.cur_freezing_ratio
                            - additional_freezing_ratio
                            + self.alpha
                        )

                        if freezing_ratio < 0:
                            freezing_ratio = 0

                        if freezing_ratio > 1:
                            alpha_readjust = False
                            freezing_ratio = self.cur_freezing_ratio

                        prev_ratio = self.cur_freezing_ratio
                        self.cur_freezing_ratio = freezing_ratio
                        alpha_readjust_expected_time = desired_time_per_epoch

                        vprint(f"Client freezing ratio: {freezing_ratio:.3f}", 2)

                        self._freezing(freezing_ratio, prev_ratio)
                    else:
                        continue

            else:
                if (
                    round_time * 0.9 > epoch_elapsed_time
                    and num_extra_epoch < self.server_config.num_extra_epoch
                ):
                    vprint(f"Client has time left for training more epochs", 2)
                    num_extra_epoch += 1
                    continue
                else:
                    break

        return trained_epochs

    async def _train_glue_dataset(self, params: dict):
        raise NotImplementedError(
            "_train_glue_dataset function is not implemented for FitFL"
        )

    async def train(self, params: dict = None):
        if self.server_config.dataset in [
            "cola",
            "sst2",
            "mrpc",
            "sts-b",
            "qqp",
            "mnli",
            "mnli-mm",
            "qnli",
            "rte",
            "wnli",
        ]:
            return await self._train_glue_dataset(params)
        else:
            return await self._train_default_dataset(params)
