from typing import TYPE_CHECKING, Callable

import dill

from .fl_handler import FLHandler

if TYPE_CHECKING:
    from server_args import Config

    from ...global_dict.global_dict import GlobalDict


class NestFLHandler(FLHandler):
    def __init__(
        self,
        config: "Config",
        global_dict: "GlobalDict",
    ):
        super().__init__(config, global_dict)

    def get_all_handler(self) -> dict[str, Callable]:
        return {
            "request_server_config": self._request_server_config,
            "client_information": self._client_information,
            "wait_for_training": self._wait_for_training,
            "submit_model": self._submit_model,
        }
