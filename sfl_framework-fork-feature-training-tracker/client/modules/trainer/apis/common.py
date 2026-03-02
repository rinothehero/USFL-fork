import pickle
from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from modules.ws.connection import Connection


def _parse_message(message):
    """Parse message from connection: dict (InMemory) or string (WebSocket)."""
    if isinstance(message, dict):
        return message
    return orjson.loads(message)


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

        message = await self.connection.receive_message()
        return _parse_message(message)["data"]

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
            parsed = _parse_message(message)

            if parsed["event"] == "start_round":
                model_raw = parsed["params"]["model"]
                if isinstance(model_raw, str):
                    # WebSocket path: hex-encoded pickle
                    global_model = pickle.loads(bytes.fromhex(model_raw))
                else:
                    # InMemory direct path: already a Python object
                    global_model = model_raw

                return (
                    global_model,
                    parsed["params"]["training_params"],
                )
            elif parsed["event"] == "kill_round":
                return (None, None)

    async def wait_for_gradients(self):
        while True:
            message = await self.connection.receive_message()
            parsed = _parse_message(message)

            if parsed["event"] == "gradients":
                grads_raw = parsed["params"]["gradients"]
                if isinstance(grads_raw, str):
                    # WebSocket path: hex-encoded pickle
                    gradients = pickle.loads(bytes.fromhex(grads_raw))
                else:
                    # InMemory direct path: already a tensor/tuple
                    gradients = grads_raw

                return (
                    gradients,
                    parsed["params"]["model_index"],
                )

    async def submit_activations(self, activations: dict, signiture: str):
        await self.connection.send_json(
            {
                "event": "submit_activations",
                "params": {
                    "activations": activations,  # Direct dict (no pickle+hex)
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
                    "model": model,  # Direct object (no pickle+hex)
                    "signiture": signiture,
                    **params,
                },
            }
        )
