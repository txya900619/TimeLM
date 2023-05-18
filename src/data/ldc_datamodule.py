import os
from glob import glob
from typing import Any, Dict, Optional, Tuple

import sentencepiece as spm
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


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
    def __init__(self, data_paths: str, tokenizer_model_path: str):
        tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
        transcriptions = []

        for data_path in data_paths:
            with open(data_path, encoding="utf-8") as data:
                transcriptions += data.readlines()
                data.close()

        tokens_list = [
            tokenizer.EncodeAsIds(transcription.strip()) for transcription in transcriptions
        ]
        self.lengths = torch.as_tensor([len(tokens) + 1 for tokens in tokens_list])
        self.tokens_bos = pad_sequence(
            [torch.LongTensor([tokenizer.bos_id()] + tokens) for tokens in tokens_list],
            batch_first=True,
        )
        self.tokens_eos = pad_sequence(
            [torch.LongTensor(tokens + [tokenizer.eos_id()]) for tokens in tokens_list],
            batch_first=True,
        )

    def __len__(self):
        return self.tokens_bos.shape[0]

    def __getitem__(self, index):
        return self._unpack_pad(self.tokens_bos[index], self.lengths[index]), self._unpack_pad(
            self.tokens_eos[index], self.lengths[index]
        )

    def _unpack_pad(self, padded: torch.Tensor, length: int):
        max_length = padded.shape[0]
        idx = torch.arange(max_length)

        mask = idx < length
        unpacked = padded[mask]
        return unpacked


class LDCDataModule(LightningDataModule):
    def __init__(
        self,
        data_folder_path: str,
        tokenizer_model_path: str,
        train_val_test_split: Tuple[float, float, float] = [0.8, 0.1, 0.1],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_folder_path = data_folder_path
        self.tokenizer_model_path = tokenizer_model_path
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: str) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            data_paths = glob(os.path.join(self.data_folder_path, "*.txt"))

            self.data_train, self.data_val, self.data_test = random_split(
                LDCDataset(data_paths, self.tokenizer_model_path),
                lengths=[int(len(data_paths) * split) for split in self.train_val_test_split],
            )

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
