from typing import TYPE_CHECKING, Tuple

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .base_dataset import BaseDataset
from .maskable_dataset import MaskableDataset

if TYPE_CHECKING:
    from client_args import ServerConfig


class GLUEDataset(BaseDataset):
    def __init__(
        self,
        config: "ServerConfig",
    ) -> None:
        self.initialized = False
        self.config = config
        self.trainset: MaskableDataset | None = None
        self.trainloader: DataLoader | None = None
        self._loader_seed = self._build_loader_seed()
        self._loader_generator = torch.Generator()
        self._loader_generator.manual_seed(self._loader_seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.glue_tokenizer)
        self.sentence_keys: dict[str, tuple[str, str | None]] = {
            "cola": ("sentence", None),
            "sst2": ("sentence", None),
            "mrpc": ("sentence1", "sentence2"),
            "sts-b": ("sentence1", "sentence2"),
            "qqp": ("question1", "question2"),
            "mnli": ("premise", "hypothesis"),
            "mnli-mm": ("premise", "hypothesis"),
            "qnli": ("question", "sentence"),
            "rte": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
            "ax": ("premise", "hypothesis"),
        }

        if self.config.dataset not in self.sentence_keys:
            raise ValueError(f"Unknown dataset_name {self.config.dataset}")

        self.sentence1_key, self.sentence2_key = self.sentence_keys[self.config.dataset]

    def _build_loader_seed(self) -> int:
        base_seed = int(getattr(self.config, "seed", 0))
        mask_ids = list(getattr(self.config, "mask_ids", []) or [])
        checksum = 0
        for idx, value in enumerate(mask_ids[:2048]):
            checksum = (checksum + (idx + 1) * int(value)) % 2147483647
        return (base_seed + checksum) % 2147483647

    def _download_dataset(self) -> Dataset:
        dataset = load_dataset(
            "glue", self.config.dataset, cache_dir=self.config.dataset_path
        )
        return dataset

    def _tokenize_function(self, examples):
        if self.sentence2_key is None:
            return self.tokenizer(
                examples[self.sentence1_key],
                padding="max_length",
                truncation=True,
                max_length=self.config.glue_max_seq_length,
            )
        else:
            return self.tokenizer(
                examples[self.sentence1_key],
                examples[self.sentence2_key],
                padding="max_length",
                truncation=True,
                max_length=self.config.glue_max_seq_length,
            )

    def _prepare_dataset(self, dataset) -> Dataset:
        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)

        if self.config.dataset == "sts-b":

            def _cast_label_to_float(examples):
                examples["label"] = [float(label) for label in examples["label"]]
                return examples

            tokenized_dataset = tokenized_dataset.map(_cast_label_to_float)

        columns = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in tokenized_dataset["train"].features:
            columns.append("token_type_ids")

        tokenized_dataset.set_format("torch", columns=columns)
        return tokenized_dataset

    def _create_loader(self) -> DataLoader:
        train_dataset = self.trainset

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=self._loader_generator,
            drop_last=True,  # For BatchNorm
        )

        return train_loader

    def update_batch_size(self, batch_size: int) -> None:
        self.config.batch_size = batch_size
        self.trainloader = self._create_loader()

    # It limits the dataset amount per each label
    def update_amount_per_label(self, amount_limit_per_label: dict[int, int]) -> None:
        self.trainset.update_amount_per_label(amount_limit_per_label)
        self.trainloader = self._create_loader()

    def reshuffle_dataset(self) -> None:
        self.trainset.update_limited_indices()
        self.trainloader = self._create_loader()

    def initialize(self) -> None:
        dataset = self._download_dataset()
        tokenized_dataset = self._prepare_dataset(dataset)

        self.trainset = MaskableDataset(
            tokenized_dataset["train"], self.config.mask_ids
        )
        self.trainloader = self._create_loader()

        self.initialized = True

    def get_trainset(self) -> Dataset:
        return self.trainset

    def get_trainloader(self) -> DataLoader:
        return self.trainloader

    def get_num_classes(self) -> int:
        if self.config.dataset in [
            "cola",
            "sst2",
            "mrpc",
            "qqp",
            "qnli",
            "rte",
            "wnli",
        ]:
            return 2
        elif self.config.dataset in ["mnli", "mnli-mm"]:
            return 3
        elif self.config.dataset == "sts-b":
            return 1
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
