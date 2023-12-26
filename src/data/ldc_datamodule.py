import json
import os
import random
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sentencepiece as spm
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


def collate_fn(batch):
    tokens_bos_list = []
    tokens_eos_list = []
    for tokens_bos, tokens_eos in batch:
        tokens_bos_list.append(tokens_bos)
        tokens_eos_list.append(tokens_eos)
    return (
        pad_sequence(tokens_bos_list, batch_first=True),
        pad_sequence(tokens_eos_list, batch_first=True),
    )


class LDCDataset(Dataset):
    def __init__(self, json_list_file_path: str, bos: int, eos: int):
        self.json_list_file_path = json_list_file_path
        self.bos = bos
        self.eos = eos

        offsets = []
        with open(json_list_file_path, "rb") as json_list_file:
            while True:
                offset = json_list_file.tell()
                if not json_list_file.readline():
                    break
                offsets.append(offset)
            json_list_file.close()
        self.offsets = np.array(offsets)

    def __len__(self):
        return self.offsets.size

    def __getitem__(self, index):
        offset = self.offsets[index]
        with open(self.json_list_file_path, encoding="utf-8") as data_file:
            data_file.seek(offset)
            json_string = data_file.readline()
            data_file.close()
            _, tokens = json.loads(json_string.strip())  # _ is date string ex. 19910101
            tokens_bos = torch.LongTensor([self.bos] + tokens)
            tokens_eos = torch.LongTensor(tokens + [self.eos])
            return (tokens_bos, tokens_eos)


class LDCDataModule(LightningDataModule):
    def __init__(
        self,
        data_folder_path: str,
        tokenizer_model_path: str,
        glob_match_string: str,
        tmp_dir: str,
        per_artical: bool = False,
        train_val_test_split: Tuple[float, float, float] = [0.8, 0.1, 0.1],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_folder_path = data_folder_path
        self.tokenizer_model_path = tokenizer_model_path
        self.glob_match_string = glob_match_string
        self.tmp_dir = tmp_dir
        self.per_artical = per_artical
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_json_list_file_path = os.path.join(
            self.tmp_dir, f"{os.path.basename(self.data_folder_path)}_train.txt"
        )
        self.valid_json_list_file_path = os.path.join(
            self.tmp_dir, f"{os.path.basename(self.data_folder_path)}_valid.txt"
        )
        self.test_json_list_file_path = os.path.join(
            self.tmp_dir, f"{os.path.basename(self.data_folder_path)}_test.txt"
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

        data_paths = glob(os.path.join(self.data_folder_path, self.glob_match_string))
        train_tokens_json_list = []
        valid_tokens_json_list = []
        test_tokens_json_list = []

        for data_path in data_paths:
            date = os.path.splitext(os.path.basename(data_path))[0]
            if self.per_artical:
                rand_tokens_json_list: List[Any] = random.choices(  # nosec
                    [train_tokens_json_list, valid_tokens_json_list, test_tokens_json_list],
                    weights=self.train_val_test_split,
                )[0]
            with open(data_path, encoding="utf-8") as data:
                for line in data.readlines():
                    if not self.per_artical:
                        rand_tokens_json_list: List[Any] = random.choices(  # nosec
                            [
                                train_tokens_json_list,
                                valid_tokens_json_list,
                                test_tokens_json_list,
                            ],
                            weights=self.train_val_test_split,
                        )[0]
                    rand_tokens_json_list.append(
                        json.dumps([date, tokenizer.EncodeAsIds(line.strip())])
                    )
                data.close()

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

            self.data_train = LDCDataset(self.train_json_list_file_path, self.bos, self.eos)
            self.data_val = LDCDataset(self.valid_json_list_file_path, self.bos, self.eos)
            self.data_test = LDCDataset(self.test_json_list_file_path, self.bos, self.eos)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
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
