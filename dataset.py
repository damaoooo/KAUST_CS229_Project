import pickle
import random
from typing import Union

import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


class SCDEDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, item):
        r1, r2, label = self.data[item]

        r1 = self._to_tensor(r1)
        r2 = self._to_tensor(r2)

        label = torch.tensor(label)

        return r1['input_ids'], r1['token_type_ids'], r1['attention_mask'], r2['input_ids'], r2['token_type_ids'], r2['attention_mask'], label

    def _to_tensor(self, sample):
        sample['input_ids'] = torch.LongTensor(sample['input_ids'])
        sample['token_type_ids'] = torch.LongTensor(sample['token_type_ids'])
        sample['attention_mask'] = torch.LongTensor(sample['attention_mask'])
        return sample

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    batch_size = len(batch[0])
    random_idx = list(range(4 * batch_size))
    random.shuffle(random_idx)
    r1, r2, w1, w2 = batch
    input_ids = torch.stack([r1['input_ids'], r2['input_ids'], w1['input_ids'], w2['input_ids']],
                            dim=0)[random_idx]
    token_types = torch.stack([r1['token_type_ids'], r2['token_type_ids'], w1['token_type_ids'], w2['token_type_ids']],
                              dim=0)[random_idx]
    attn_mask = torch.stack([r1['attention_mask'], r2['attention_mask'], w1['attention_mask'], w2['attention_mask']],
                            dim=0)[random_idx]
    label = torch.tensor([1] * (2 * batch_size) + [0] * (2 * batch_size))
    return input_ids, token_types, attn_mask, label


class SCDEDataModule(pl.LightningDataModule):
    def __init__(self, data_file: str = "./data.cache", batch_size: int = 16, num_workers: int = 8):
        super().__init__()
        self.data_file = data_file
        self.train_set: Union[Dataset, None] = None
        self.val_set: Union[Dataset, None] = None
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        with open(self.data_file, "rb") as f:
            cache = pickle.load(f)
        train_data, val_data = cache['train'], cache['dev']
        dataset = SCDEDataset(train_data + val_data)
        

        self.train_set, self.val_set = random_split(dataset, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    data_module = SCDEDataModule()
    data_module.prepare_data()
    train_set = data_module.train_dataloader()
    cnt = 0
    for i in iter(train_set):
        cnt += 1
    print(cnt)
