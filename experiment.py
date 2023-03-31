from utils import *
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
import torch
from copy import deepcopy
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

data = read_dataset("./CLOTH")

dist = []
for cloth in data:
    tk = tokenizer(cloth.article, return_tensors="pt", max_length=512)['input_ids'].shape[1]
    dist.append(tk)

plt.hist(dist)
plt.show()