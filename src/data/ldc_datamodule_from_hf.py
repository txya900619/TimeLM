import json
import os
import re
from typing import Any, Dict, Optional

import sentencepiece as spm
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class LDCDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        tokenizer_model_path: str,
        glob_match_string: str,
        tmp_dir: str,
        dataset_class: type[Dataset],
        date_text: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        test_dataset_class: Optional[type[Dataset]] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.dataset_name = dataset_name
        self.tokenizer_model_path = tokenizer_model_path
        self.glob_match_string = glob_match_string
        self.tmp_dir = tmp_dir
        self.dataset_class = dataset_class
        self.test_dataset_class = test_dataset_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.date_text = date_text

        self.train_json_list_file_path = os.path.join(
            self.tmp_dir, f"{os.path.basename(self.dataset_name)}_train.txt"
        )
        self.valid_json_list_file_path = os.path.join(
            self.tmp_dir, f"{os.path.basename(self.dataset_name)}_valid.txt"
        )
        self.test_json_list_file_path = os.path.join(
            self.tmp_dir, f"{os.path.basename(self.dataset_name)}_test.txt"
        )
        self.bos: Optional[int] = None
        self.eos: Optional[int] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # TODO: Add date information
        tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_model_path)

        self.bos = tokenizer.bos_id()
        self.eos = tokenizer.eos_id()

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        if os.path.exists(self.train_json_list_file_path):
            return

        hf_dataset = load_dataset(self.dataset_name)

        hf_dataset = hf_dataset.filter(
            lambda x: re.match(self.glob_match_string, x["date"]) is not None
        )

        train_tokens_json_list = []
        valid_tokens_json_list = []
        test_tokens_json_list = []

        for split, dataset in hf_dataset.items():
            if split == "train":
                tokens_json_list = train_tokens_json_list
            elif split == "validation":
                tokens_json_list = valid_tokens_json_list
            elif split == "test":
                tokens_json_list = test_tokens_json_list
            else:
                continue

            for item in dataset:
                date = item["date"]
                if self.date_text:
                    num_to_han = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
                    date = "".join([num_to_han[int(d)] for d in date])
                    date = tokenizer.EncodeAsIds(date.strip())[1:]
                tokens_json_list.append(
                    json.dumps([date, tokenizer.EncodeAsIds(item["text"].strip())])
                )

        with open(self.train_json_list_file_path, "w", encoding="utf-8") as train_json_list_file:
            train_json_list_file.write("\n".join(train_tokens_json_list))
            train_json_list_file.close()

        with open(self.valid_json_list_file_path, "w", encoding="utf-8") as valid_json_list_file:
            valid_json_list_file.write("\n".join(valid_tokens_json_list))
            valid_json_list_file.close()

        with open(self.test_json_list_file_path, "w", encoding="utf-8") as test_json_list_file:
            test_json_list_file.write("\n".join(test_tokens_json_list))
            test_json_list_file.close()

    def setup(self, stage: str) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_model_path)
            self.bos = tokenizer.bos_id()
            self.eos = tokenizer.eos_id()

            self.data_train = self.dataset_class(
                self.train_json_list_file_path, self.bos, self.eos
            )
            test_dataset_class = self.test_dataset_class or self.dataset_class
            self.data_val = test_dataset_class(self.valid_json_list_file_path, self.bos, self.eos)
            self.data_test = test_dataset_class(self.test_json_list_file_path, self.bos, self.eos)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.data_train.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.data_val.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.data_test.collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = LDCDataModule()
