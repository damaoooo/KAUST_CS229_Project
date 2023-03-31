import json
import os
from dataclasses import dataclass, field
from typing import List, Union


@dataclass()
class Cloth:
    options: List[List[str]] = field(default_factory=list)
    article: str = field(default="")
    answer: List[int] = field(default_factory=list)


def file_reader(file_path: str):
    with open(file_path, 'r') as f:
        content = f.read()
        f.close()
    cloth = Cloth()
    content = json.loads(content)
    cloth.article = content['article'].lower()
    cloth.article = cloth.article.replace(" _ ", "[MASK]")
    cloth.options = content['options']
    content['answers'] = [ord(x) - ord('A') for x in content['answers']]
    cloth.answer = content['answers']
    return cloth


def read_dataset(directory: str, separate: bool = False) -> Union[List[Cloth], List[List[Cloth]]]:
    three_way = ["train", "test", "valid"]
    middle: List[Cloth] = []
    high: List[Cloth] = []
    for direct in three_way:
        for direct2 in ["high", "middle"]:
            temp_path = os.path.join(directory, direct, direct2)
            files = os.listdir(temp_path)
            for file in files:
                cloth = file_reader(os.path.join(temp_path, file))
                if direct2 == "high":
                    high.append(cloth)
                else:
                    middle.append(cloth)
    if separate:
        return [high, middle]
    else:
        return high + middle

