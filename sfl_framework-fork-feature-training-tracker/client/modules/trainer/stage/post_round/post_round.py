from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.trainer.apis.common import CommonAPI
    from torch.nn import Module


class PostRound:
    def __init__(self, config: "Config", server_config: "ServerConfig"):
        self.config = config
        self.server_config = server_config

    async def submit_model(
        self,
        api: "CommonAPI",
        model: "Module",
        signiture: str,
        params: dict,
    ):
        await api.submit_model(model, signiture, params)
