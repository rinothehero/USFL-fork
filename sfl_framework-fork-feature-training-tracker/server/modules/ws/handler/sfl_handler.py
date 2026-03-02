import pickle
import time
from typing import TYPE_CHECKING, Callable

from .fl_handler import FLHandler
from utils.log_utils import vprint

if TYPE_CHECKING:
    from server_args import Config

    from ....modules.global_dict.global_dict import GlobalDict


class SFLHandler(FLHandler):
    def __init__(
        self,
        config: "Config",
        global_dict: "GlobalDict",
    ):
        super().__init__(config, global_dict)

    async def _submit_activations(self, params: dict):
        signiture = params["signiture"]
        model_queue = self.global_dict.get("model_queue")

        if signiture != model_queue.signiture:
            vprint(f"Signiture mismatch: {signiture} != {model_queue.signiture}", 0)
            return {"event": "submit_activations", "data": {}}

        activations_raw = params["activations"]
        if isinstance(activations_raw, str):
            # WebSocket path: hex-encoded pickle
            activations = pickle.loads(bytes.fromhex(activations_raw))
        else:
            # InMemory direct path: already a Python dict
            activations = activations_raw
        self.global_dict.get("activation_queue").append(activations)

        return None

    def get_all_handler(self) -> dict[str, Callable]:
        return {
            "request_server_config": self._request_server_config,
            "wait_for_training": self._wait_for_training,
            "client_information": self._client_information,
            "submit_model": self._submit_model,
            "submit_activations": self._submit_activations,
        }
