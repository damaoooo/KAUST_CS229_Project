import os.path
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import json
import pickle
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import dataset, dataloader, random_split
import lightning.pytorch as pl
from transformers import AutoTokenizer

from utils import *


class CLOTHDataset(dataset.Dataset):
    def __init__(self, data: List[Cloth], tokenizer_name: str, max_length: int = 512):
        super().__init__()
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_id = self.tokenizer("[MASK]")["input_ids"][1]
        self.max_length = max_length
        self.data = self.transform()

    def transform(self):
        result: List[Dict] = []
        for cloth in self.data:
            cloth: Cloth
            if not cloth.options:
                continue
            p = {"options": [], "article": "", "answer": []}
            article = self.tokenizer(cloth.article, padding="max_length", max_length=self.max_length,
                                     return_tensors="pt")

            article, mask = article['input_ids'][0], article["attention_mask"][0]

            if len(article) > self.max_length:
                continue

            length = len(cloth.options)
            position = np.argwhere(article.numpy() == self.mask_id).reshape(-1)
            position = pad_sequence([torch.tensor(position, dtype=torch.long), torch.zeros([20])], batch_first=True)[0]
            options = torch.stack(
                [self.tokenizer(x, return_tensors="pt", padding=True)['input_ids'][:, 1] for x in cloth.options]
            )
            options = pad_sequence([options, torch.zeros(20, 4)], batch_first=True)[0]
            answer = torch.tensor(cloth.answer, dtype=torch.long)
            answer = pad_sequence([answer, torch.zeros([20])], batch_first=True)[0]

            p["attn_mask"] = mask
            p["length"] = length
            p["answer"] = answer
            p["article"] = article
            p["options"] = options
            p["position"] = position

            result.append(p.copy())
        return result

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class CLOTHDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "./CLOTH", batch_size: int = 32, use_cache: str = "./.data_cache",
                 tokenizer: str = "bert-base-uncased", max_length: int = 512, separate: bool = False, num_workers=8):
        super().__init__()
        self.num_workers = num_workers
        self.train_set: Union[dataset.Dataset, None] = None
        self.val_set: Union[dataset.Dataset, None] = None
        self.data_path = data_path
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.tokenizer_name = tokenizer
        self.max_length = max_length
        self.separate = separate

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.tokenizer_name)

        if not os.path.exists(self.data_path) and (self.use_cache != "" and not os.path.exists(self.use_cache)):
            raise FileExistsError("The data set and cache are both not exist, please provide a valid one")

        if self.use_cache:
            if os.path.exists(self.use_cache):
                with open(self.use_cache, 'rb') as f:
                    data = pickle.load(f)
                    f.close()
            else:
                data = read_dataset(self.data_path, separate=self.separate)
                with open(self.use_cache, 'wb') as f:
                    pickle.dump(data, f)
                    f.close()
        else:
            data = read_dataset(self.data_path, separate=self.separate)

        if self.separate:
            data = data[self.separate-1]

        total_dataset = CLOTHDataset(data, tokenizer_name=self.tokenizer_name, max_length=self.max_length)
        self.train_set, val_set = random_split(total_dataset, [0.8, 0.2])

    def train_dataloader(self):
        return dataloader.DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers)

    def val_dataloader(self):
        return dataloader.DataLoader(dataset=self.val_set, batch_size=self.batch_size,
                                     num_workers=self.num_workers)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    p = CLOTHDataModule()
    p.prepare_data()
    train = p.train_dataloader()
    idx = 0
    for i in train:
        print(idx)
        idx += 1



