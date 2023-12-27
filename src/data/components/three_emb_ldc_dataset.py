import json
import random
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ThreeEmbLDCDataset(Dataset):
    def __init__(self, json_list_file_path: str, bos: int, eos: int):
        self.json_list_file_path = json_list_file_path
        self.bos = bos
        self.eos = eos
        self.no_date_prob = 0.0

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
            year_token = 0
            month_token = 0
            day_token = 0

            if random.random() >= self.no_date_prob:  # nosec
                year_token = int(date[:4]) - 1990
                month_token = int(date[4:6])
                day_token = int(date[6:])

            tokens_bos = torch.LongTensor([self.bos] + tokens)
            tokens_eos = torch.LongTensor(tokens + [self.eos])
            return (tokens_bos, tokens_eos, year_token, month_token, day_token)

    def collate_fn(self, batch: Tuple[Tensor, Tensor, int, int, int]):
        tokens_bos_list = []
        tokens_eos_list = []
        year_token_list = []
        month_token_list = []
        day_token_list = []
        for tokens_bos, tokens_eos, year_token, month_token, day_token in batch:
            tokens_bos_list.append(tokens_bos)
            tokens_eos_list.append(tokens_eos)
            year_token_list.append(year_token)
            month_token_list.append(month_token)
            day_token_list.append(day_token)
        return (
            pad_sequence(tokens_bos_list, batch_first=True),
            pad_sequence(tokens_eos_list, batch_first=True),
            torch.LongTensor(year_token_list),
            torch.LongTensor(month_token_list),
            torch.LongTensor(day_token_list),
        )
