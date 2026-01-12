import pickle
from typing import TYPE_CHECKING, Callable

from .base_handler import BaseHandler

if TYPE_CHECKING:
    from server_args import Config

    from ...global_dict.global_dict import GlobalDict


class FLHandler(BaseHandler):
    def __init__(
        self,
        config: "Config",
        global_dict: "GlobalDict",
    ):
        self.config = config
        self.global_dict = global_dict

    async def _request_server_config(self, params: dict):
        configs = self.config.__dict__
        client_masks = self.global_dict.get("client_data_masks")

        configs["mask_ids"] = client_masks[int(params["client_id"])]
        return {"event": "request_server_config", "data": configs}

    async def _wait_for_training(self, params: dict):
        self.global_dict.set_waiting_clients(params["client_id"], True)
        return None

    async def _submit_model(self, params: dict):
        model = pickle.loads(bytes.fromhex(params["model"]))
        model_size = len(params["model"])
        signiture = params["signiture"]

        for key in ["model", "signiture"]:
            params.pop(key)
        
        # Filter out gradient-related fields to prevent JSON bloat
        filtered_params = {k: v for k, v in params.items() 
                          if k not in ['measurement_gradient', 'client_gradients', 'gradient_weights', 'client_gradient']}

        model_queue = self.global_dict.get("model_queue")
        result = model_queue.add_model(params["client_id"], model, signiture, params)

        if result:
            self.global_dict.add_event(
                "MODEL_RECIEVED",
                {
                    "client_id": params["client_id"],
                    "size": model_size,
                    **filtered_params,
                },
            )

        return None

    async def _client_information(self, params):
        client_id = int(params["client_id"])

        client_informations = {}

        try:
            client_informations = self.global_dict.get("client_informations")
        except:
            pass

        client_informations[client_id] = params
        self.global_dict.set("client_informations", client_informations)
        return None

    def get_all_handler(self) -> dict[str, Callable]:
        return {
            "request_server_config": self._request_server_config,
            "wait_for_training": self._wait_for_training,
            "submit_model": self._submit_model,
            "client_information": self._client_information,
        }
