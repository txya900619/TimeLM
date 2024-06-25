import json
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class DateTextLDCDataset(Dataset):
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

    def __getitem__(self, index: int):
        offset = self.offsets[index]
        with open(self.json_list_file_path, encoding="utf-8") as data_file:
            data_file.seek(offset)
            json_string = data_file.readline()
            data_file.close()
            date, tokens = json.loads(json_string.strip())  # _ is date string ex. 19910101
            date_tokens = torch.LongTensor(date)
            tokens_bos = torch.LongTensor([self.bos] + tokens)
            tokens_eos = torch.LongTensor(tokens + [self.eos])
            return (tokens_bos, tokens_eos, date_tokens)

    def collate_fn(self, batch: Tuple[Tensor, Tensor]):
        tokens_bos_list = []
        tokens_eos_list = []
        date_tokens_list = []
        for tokens_bos, tokens_eos, date_tokens in batch:
            tokens_bos_list.append(tokens_bos)
            tokens_eos_list.append(tokens_eos)
            date_tokens_list.append(date_tokens)
        return (
            pad_sequence(tokens_bos_list, batch_first=True),
            pad_sequence(tokens_eos_list, batch_first=True),
            torch.stack(date_tokens_list),
        )
