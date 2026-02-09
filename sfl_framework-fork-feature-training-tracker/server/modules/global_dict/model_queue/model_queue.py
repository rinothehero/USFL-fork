import hashlib
import random
from collections import deque
from typing import TYPE_CHECKING

from utils.log_utils import vprint

if TYPE_CHECKING:
    from torch.nn import Module


class ModelQueue:
    def __init__(self):
        self.signiture = ""
        self.queue = deque()
        self.insert_mode = False

    def _generate_signiture(self):
        self.signiture = hashlib.sha256(random.randbytes(1024)).hexdigest()

    def get_signiture(self):
        return self.signiture

    def start_insert_mode(self):
        self._generate_signiture()
        self.insert_mode = True

    def end_insert_mode(self):
        self.insert_mode = False

    def clear(self):
        self.queue.clear()

    def add_model(
        self,
        id: int,
        model: "Module",
        signiture: str,
        params: dict,
    ):
        if self.signiture != signiture:
            vprint(f"Signiture mismatch: {self.signiture} != {signiture} (late model)", 0)
            return

        vprint(f"client {id} submitted model", 2)

        if self.insert_mode:
            self.queue.append([id, model, params])
            return True
        else:
            vprint(f"Model queue is not in insert mode (late model)", 0)
            return False

    def get_model(self):
        if len(self.queue) == 0:
            return None
        id, model, dataset_size = self.queue.popleft()
        return id, model, dataset_size

    def get_all_models(self):
        ids = []
        models = []
        params = []

        while len(self.queue) != 0:
            id, model, param = self.queue.popleft()
            ids.append(id)
            models.append(model)
            params.append(param)

        return ids, models, params

    def get_queue_size(self):
        return len(self.queue)
