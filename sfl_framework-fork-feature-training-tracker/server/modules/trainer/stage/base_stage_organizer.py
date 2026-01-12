from abc import ABC, abstractmethod


class BaseStageOrganizer(ABC):
    def _count_parameters(self, model):
        total_parameters = 0
        for param in model.parameters():
            total_parameters += param.nelement()
        return total_parameters

    @abstractmethod
    async def _pre_round(self, round_number: int):
        pass

    @abstractmethod
    async def _in_round(self, round_number: int):
        pass

    @abstractmethod
    async def _post_round(self, round_number: int):
        pass

    async def run_pre_round(self, round_number: int):
        print(f"[Round {round_number}] Pre-round")
        await self._pre_round(round_number)

    async def run_in_round(self, round_number: int):
        print(f"[Round {round_number}] In-round")
        await self._in_round(round_number)

    async def run_post_round(self, round_number: int):
        print(f"[Round {round_number}] Post-round")
        await self._post_round(round_number)