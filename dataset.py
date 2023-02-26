import math
import os.path
import pickle

import torch
from torch.utils.data import dataset, dataloader, random_split

from transformers import AutoTokenizer
import pytorch_lightning as pl
from utils import *

import matplotlib.pyplot as plt

overlaps = []

class TOEFLDataset(dataset.Dataset):
    def __init__(self, data: List[Tpo], tokenizer_name: str, length: int = 300, windows: int = 4):
        super().__init__()
        self.data: List[Tpo] = data
        self.slice_length = length
        self.windows = windows
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.data = self.transform()


    def transform(self):
        res = []
        for i in self.data:
            i: Tpo
            p = {"corpus": [], "answer": -1}
            if i.is_multi or i.answer == -1:
                continue

            # length <= slice + (window-1) * (slice - overlap)
            # slice - overlap >= (length - slice) / (window - 1)
            # overlap <= slice -  (length - slice) / (window - 1)
            overlap = math.floor(self.slice_length - (len(i.sentence) - self.slice_length + self.windows) / (self.windows - 1))
            passage_slices = slicing(i.sentence, self.slice_length, overlap)
            overlaps.append(overlap)
            for op in i.options:
                for passage_slice in passage_slices:
                    construct = "Question: {}. Passage: {}, Option: {}".format(
                        ' '.join(i.question), ' '.join(passage_slice), ' '.join(op))
                    p["corpus"].append(construct)
            p["answer"] = i.answer
            try:
                assert len(p['corpus']) == 4 * self.windows
            except AssertionError:
                print("length=",len(p['corpus']), "length_passage=", len(i.sentence), "overlap=", overlap)
                continue
            res.append(self.tokenize(p))
        return res

    def tokenize(self, corpus: dict):
        corpus_ = self.tokenizer(corpus["corpus"], padding="max_length", max_length=512, return_tensors='pt')
        answer = torch.tensor(corpus["answer"], dtype=torch.long)
        return {"corpus": corpus_['input_ids'], "mask": corpus_["attention_mask"], "answer": answer}

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class TOEFLDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./TOEFL-QA/data", batch_size=32, use_cache="./data_cache", tokenizer: str = "bert-base-cased", windows=4):
        super().__init__()
        self.test_set: Union[dataset.Dataset, None] = None
        self.val_set: Union[dataset.Dataset, None] = None
        self.windows = windows
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.tokenizer_name = tokenizer
        # TODO: Use cache instead of new data

    def load_dataset(self, path: str):
        d = []
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as f:
                p = f.read()
                f.close()
            d.append(parse_text(p, file))
        return d

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.tokenizer_name)

        if not os.path.exists(self.data_dir):
            raise FileExistsError("No, Please Provide a valid dataset position")

        if self.use_cache:
            if os.path.exists(self.use_cache):
                with open(self.use_cache, "rb") as f:
                    total_dataset = pickle.load(f)
                    f.close()
            else:
                total_dataset = self.load_dataset(self.data_dir)
                if self.use_cache:
                    with open(self.use_cache, 'wb') as f:
                        pickle.dump(total_dataset, f)
                        f.close()
        else:
            total_dataset = self.load_dataset(self.data_dir)

        total_dataset = TOEFLDataset(total_dataset, tokenizer_name=self.tokenizer_name, windows=self.windows)
        train_size = int(0.8 * len(total_dataset))
        val_size = len(total_dataset) - train_size
        self.train_set, self.val_set = random_split(total_dataset, [train_size, val_size])

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return dataloader.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return dataloader.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4)


if __name__ == "__main__":
    p = TOEFLDataModule(use_cache="")
    p.prepare_data()
    p.setup(stage='fit')
    train = p.train_dataloader()
    plt.boxplot(overlaps)
    plt.show()
    for i in train:
        print(i)
        break
