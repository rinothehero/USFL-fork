import asyncio
import base64
import pickle
import time
from typing import TYPE_CHECKING

import orjson

from utils.log_utils import vprint

from ....model.base_model import BaseModel
from ....ws.connection import Connection
from ...seletor.base_selector import BaseSelector

if TYPE_CHECKING:
    import torch
    from server_args import Config

    from ....global_dict.global_dict import GlobalDict


class PreRound:
    def __init__(self, config: "Config", global_dict: "GlobalDict"):
        self.config = config
        self.global_dict = global_dict

    async def wait_for_client_informations(self):
        last_key_length = 0
        vprint(
            f"Waiting for {self.config.num_clients} clients to send client information... ({last_key_length})",
            2,
        )

        while True:
            client_informations = {}

            try:
                client_informations = self.global_dict.get("client_informations")
            except:
                pass

            key_length = len(client_informations.keys())
            if key_length >= self.config.num_clients:
                vprint(f"Received client information from {key_length} clients", 2)
                break
            else:
                if last_key_length != key_length:
                    vprint(f"Received client information from {key_length} clients", 2)

                last_key_length = key_length

            await asyncio.sleep(0.01)

    async def wait_for_clients(self):
        last_client_count = self.global_dict.get_waiting_clients_count()
        vprint(
            f"Waiting for {self.config.num_clients} clients to connect... ({last_client_count})",
            2,
        )

        while True:
            total_waiting_clients = self.global_dict.get_waiting_clients_count()

            if total_waiting_clients >= self.config.num_clients:
                vprint(f"Found {self.config.num_clients} clients", 2)
                break
            else:
                if last_client_count != total_waiting_clients:
                    vprint(
                        f"Waiting for {self.config.num_clients} clients to connect... ({total_waiting_clients})",
                        2,
                    )

                last_client_count = total_waiting_clients

            await asyncio.sleep(0.01)

    def calculate_round_end_time(self):
        round_start_time = time.time()
        return round_start_time, round_start_time + self.config.round_duration

    def select_clients(
        self, selector: "BaseSelector", connection: "Connection", data=None
    ):
        selected_clients = selector.select(
            self.config.num_clients_per_round,
            self.global_dict.get_all_waiting_clients(),
            data,
        )

        vprint(f"Selected {len(selected_clients)} clients: {selected_clients}", 1)
        return selected_clients

    async def send_customized_global_model(
        self,
        selected_clients: list[int],
        torch_models: list["torch.nn.Module"],
        connection: "Connection",
        training_params: dict,
    ):
        messages = [
            {
                "event": "start_round",
                "params": {
                    "model": torch_model.to("cpu"),  # Direct object (no pickle+hex)
                    "training_params": training_params,
                },
            }
            for torch_model in torch_models
        ]

        await connection.send_bytes_batch(
            data=messages,
            client_ids=selected_clients,
            logging=True,
            on_start=lambda size, client_id: self.global_dict.add_event(
                "SEND_MODEL_START", {"size": size, "to": client_id}
            ),
            on_end=lambda size, client_id: self.global_dict.add_event(
                "SEND_MODEL_END", {"size": size, "to": client_id}
            ),
        )

        for selected_client in selected_clients:
            self.global_dict.set_waiting_clients(selected_client, False)

    async def send_global_model(
        self,
        selected_clients: list[int],
        model: "BaseModel",
        connection: "Connection",
        training_params: dict,
    ):
        torch_model = model.get_torch_model().to("cpu")

        data_to_send = [
            {
                "event": "start_round",
                "params": {
                    "model": torch_model,  # Direct object (no pickle+hex)
                    "training_params": training_params,
                },
            }
            for _ in selected_clients
        ]

        await connection.send_bytes_batch(
            data=data_to_send,
            client_ids=selected_clients,
            logging=True,
            on_start=lambda size, client_id: self.global_dict.add_event(
                "SEND_MODEL_START", {"size": size, "to": client_id}
            ),
            on_end=lambda size, client_id: self.global_dict.add_event(
                "SEND_MODEL_END", {"size": size, "to": client_id}
            ),
        )

        for client in selected_clients:
            self.global_dict.set_waiting_clients(client, False)
