import random
from dataclasses import dataclass, field
from typing import List, Union, Dict
import re
import math
import pickle
import json
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")


@dataclass()
class Tpo:
    sentence: List[str] = field(default_factory=list)
    options: List[List[str]] = field(default_factory=list)
    question: List[str] = field(default_factory=list)
    answer: int = -1
    is_multi: bool = False
    type: str = ""


def slicing(s: list, length: int, overlap: int):
    increase = length - overlap
    how_many = math.ceil((len(s) - length) / increase)
    res = []
    for i in range(how_many + 1):
        res.append(s[i * increase:i * increase + length])
    return res


def align(s: dict):
    """

    :type s: output from tokenizer
    """
    if len(s['input_ids']) > 512:
        s['input_ids'] = [101] + s['input_ids'][-511:]
        s['token_type_ids'] = [0] + s['token_type_ids'][-511:]
        s['attention_mask'] = [1] + s['attention_mask']
    return s


def extract_feature(s: Dict):
    passage = s['passage']
    passage = " ".join(passage).replace("<<", "##{").replace(">>", "}##")

    candidates: List[str] = s['candidates']

    answer = s['answer_sequence']
    answer = [x[1] - 1 for x in answer]

    passage = re.sub(r"<\d>", "[MASK]", passage).replace("##{", "<<").replace("}##", ">>")

    sequences = passage.split("[MASK]")

    assert len(sequences) == len(answer) + 1

    slides = []

    for i in range(len(answer)):
        AP = '[MASK]'.join(sequences[: i + 1])
        AN = '[MASK]'.join(sequences[i + 1:])

        right = candidates[answer[i]]
        wrong_idx = random.choice([x for x in answer if x != answer[i]])
        wrong = candidates[wrong_idx]

        right_1 = tokenizer(AP, right, padding='max_length', max_length=512)
        right_1 = align(right_1)
        right_2 = tokenizer(right, AN, padding='max_length', max_length=512, truncation=True)

        wrong_1 = tokenizer(AP, wrong, padding='max_length', max_length=512)
        wrong_1 = align(wrong_1)
        wrong_2 = tokenizer(wrong, AN, padding='max_length', max_length=512, truncation=True)

        slides.append([right_1, right_2, 1])
        slides.append([wrong_1, wrong_2, 0])

    return slides


def read_file(data_path: str):
    dataset = []
    with open(data_path, 'r') as f:
        content = f.read()
        f.close()
    content = json.loads(content)
    for passage in content:
        dataset.extend(extract_feature(passage))
    return dataset


if __name__ == '__main__':
    train = read_file("./scde/train.json")
    dev = read_file("./scde/dev.json")
    cache = {"train": train, "dev": dev}
    with open("./data.cache", "wb") as f:
        pickle.dump(cache, f)
        f.close()

