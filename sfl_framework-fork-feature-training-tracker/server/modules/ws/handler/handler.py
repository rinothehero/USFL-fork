from typing import TYPE_CHECKING

from server_args import Config

from .fl_handler import FLHandler
from .nestfl_handler import NestFLHandler
from .scala_handler import ScalaHandler
from .sfl_handler import SFLHandler

if TYPE_CHECKING:
    from ...global_dict.global_dict import GlobalDict


def get_handler(
    config: "Config",
    global_dict: "GlobalDict",
):
    if config.method in [
        "fl",
        "fitfl",
        "fedcbs",
        "prunefl",
        "fedsparsify",
        "fedprox",
        "cl",
    ]:
        return FLHandler(config, global_dict)
    elif config.method == "nestfl":
        return NestFLHandler(config, global_dict)
    elif config.method in ["sfl", "sfl-u", "mix2sfl", "scaffold_sfl"]:
        return SFLHandler(config, global_dict)
    elif config.method in ["scala", "usfl", "sflprox"]:
        return ScalaHandler(config, global_dict)
    else:
        raise ValueError(f"Handler for method '{config.method}' not found")
