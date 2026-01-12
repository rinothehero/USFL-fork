from typing import TYPE_CHECKING, Callable

from .sfl_handler import SFLHandler

if TYPE_CHECKING:
    from server_args import Config

    from ...global_dict.global_dict import GlobalDict


class ScalaHandler(
    SFLHandler,
):
    def __init__(
        self,
        config: "Config",
        global_dict: "GlobalDict",
    ):
        SFLHandler.__init__(self, config, global_dict)

    def get_all_handler(self) -> dict[str, Callable]:
        return {
            "request_server_config": self._request_server_config,
            "wait_for_training": self._wait_for_training,
            "client_information": self._client_information,
            "submit_model": self._submit_model,
            "submit_activations": self._submit_activations,
        }
