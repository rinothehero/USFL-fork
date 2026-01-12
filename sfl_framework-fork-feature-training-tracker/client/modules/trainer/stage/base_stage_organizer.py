from abc import ABC, abstractmethod


class BaseStageOrganizer(ABC):
    @abstractmethod
    async def run_pre_round(self):
        pass

    @abstractmethod
    async def run_in_round(self):
        pass

    @abstractmethod
    async def run_post_round(self):
        pass
