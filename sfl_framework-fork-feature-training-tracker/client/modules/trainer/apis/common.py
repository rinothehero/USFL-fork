import pickle
from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from modules.ws.connection import Connection


class CommonAPI:
    def __init__(self, connection: "Connection"):
        self.connection = connection

    async def request_server_config(self):
        await self.connection.send_json(
            {
                "event": "request_server_config",
                "params": {},
            }
        )

        config = await self.connection.receive_message()

        return orjson.loads(config)["data"]

    async def notify_wait_for_training(self):
        await self.connection.send_json(
            {
                "event": "wait_for_training",
                "params": {},
            }
        )

    async def notify_client_information(self, params: dict):
        await self.connection.send_json(
            {
                "event": "client_information",
                "params": params,
            }
        )

    async def wait_for_start_round(self):
        while True:
            message = await self.connection.receive_message()
            message = orjson.loads(message)

            if message["event"] == "start_round":
                global_model = bytes.fromhex(message["params"]["model"])

                return (
                    pickle.loads(global_model),
                    message["params"]["training_params"],
                )
            elif message["event"] == "kill_round":
                return (None, None)

    async def wait_for_gradients(self):
        while True:
            message = await self.connection.receive_message()
            message = orjson.loads(message)

            if message["event"] == "gradients":
                gradients = bytes.fromhex(message["params"]["gradients"])

                return (
                    pickle.loads(gradients),
                    message["params"]["model_index"],
                )

    async def submit_activations(self, activations: dict, signiture: str):
        await self.connection.send_json(
            {
                "event": "submit_activations",
                "params": {
                    "activations": pickle.dumps(activations).hex(),
                    "signiture": signiture,
                },
            },
            logging=False,
        )

    async def submit_model(self, model, signiture, params):
        model = model.to("cpu")
        await self.connection.send_json(
            {
                "event": "submit_model",
                "params": {
                    "model": pickle.dumps(model).hex(),
                    "signiture": signiture,
                    **params,
                },
            }
        )
