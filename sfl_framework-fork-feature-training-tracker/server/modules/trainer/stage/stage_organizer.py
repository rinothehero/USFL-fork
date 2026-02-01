from typing import TYPE_CHECKING

from .cl_stage_organizer import CLStageOrganizer
from .fedsparsify_stage_organizer import FedSparsifyStageOrganizer
from .fitfl_stage_organizer import FitFLStageOrganizer
from .fl_stage_organizer import FLStageOrganizer
from .mix2sfl_stage_organizer import Mix2SFLStageOrganizer
from .nestfl_starge_organizer import NestFLStageOrganizer
from .prunefl_stage_organizer import PruneFLStageOrganizer
from .scala_stage_organizer import ScalaStageOrganizer
from .scaffold_sfl_stage_organizer import ScaffoldSFLStageOrganizer
from .sfl_stage_organizer import SFLStageOrganizer
from .sflprox_stage_organizer import SFLProxStageOrganizer
from .usfl_stage_organizer import USFLStageOrganizer

if TYPE_CHECKING:
    from server_args import Config

    from ...dataset.base_dataset import BaseDataset
    from ...global_dict.global_dict import GlobalDict
    from ...model.base_model import BaseModel
    from ...ws.connection import Connection
    from ..aggregator.base_aggregator import BaseAggregator
    from ..seletor.base_selector import BaseSelector
    from ..splitter.base_splitter import BaseSplitter


def get_stage_organizer(
    config: "Config",
    connection: "Connection",
    global_dict: "GlobalDict",
    aggregator: "BaseAggregator",
    model: "BaseModel",
    dataset: "BaseDataset",
    selector: "BaseSelector",
    splitter: "BaseSplitter",
):
    if config.method in ["fl", "fedprox", "fedcbs"]:
        return FLStageOrganizer(
            config,
            connection,
            global_dict,
            aggregator,
            model,
            dataset,
            selector,
        )
    elif config.method == "nestfl":
        return NestFLStageOrganizer(
            config,
            connection,
            global_dict,
            aggregator,
            model,
            dataset,
            selector,
        )
    elif config.method == "fitfl":
        return FitFLStageOrganizer(
            config,
            connection,
            global_dict,
            aggregator,
            model,
            dataset,
            selector,
        )
    elif config.method == "prunefl":
        return PruneFLStageOrganizer(
            config,
            connection,
            global_dict,
            aggregator,
            model,
            dataset,
            selector,
        )
    elif config.method == "fedsparsify":
        return FedSparsifyStageOrganizer(
            config,
            connection,
            global_dict,
            aggregator,
            model,
            dataset,
            selector,
        )
    elif config.method == "sfl":
        return SFLStageOrganizer(
            config,
            connection,
            global_dict,
            aggregator,
            model,
            dataset,
            selector,
            splitter,
        )
    elif config.method == "scaffold_sfl":
        return ScaffoldSFLStageOrganizer(
            config,
            connection,
            global_dict,
            aggregator,
            model,
            dataset,
            selector,
            splitter,
        )
    elif config.method == "scala":
        return ScalaStageOrganizer(
            config,
            connection,
            global_dict,
            aggregator,
            model,
            dataset,
            selector,
            splitter,
        )
    elif config.method == "usfl":
        return USFLStageOrganizer(
            config,
            connection,
            global_dict,
            aggregator,
            model,
            dataset,
            selector,
            splitter,
        )
    elif config.method == "mix2sfl":
        return Mix2SFLStageOrganizer(
            config,
            connection,
            global_dict,
            aggregator,
            model,
            dataset,
            selector,
            splitter,
        )
    elif config.method == "sflprox":
        return SFLProxStageOrganizer(
            config,
            connection,
            global_dict,
            aggregator,
            model,
            dataset,
            selector,
            splitter,
        )
    elif config.method == "cl":
        return CLStageOrganizer(
            config,
            global_dict,
            model,
            dataset,
        )

    else:
        raise ValueError(f"Invalid method: {config.method}")
