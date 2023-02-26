from dataclasses import dataclass, field
from typing import List, Union
import math


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
    for i in range(how_many+1):
        res.append(s[i*increase:i*increase + length])
    return res


def parse_text(data: str, title: str):
    lines = data.splitlines(keepends=False)

    tpo = Tpo()
    option_number = -1
    for line in lines:
        tokens = line.split(' ')
        if tokens[0] == 'SENTENCE':
            tpo.sentence.extend(tokens[1:])
        elif tokens[0] == 'QUESTION':
            tpo.question.extend(tokens[1:])
        elif tokens[0] == 'OPTION':
            option_number += 1
            tpo.options.append(tokens[1:-1])
            if tokens[-1] == '1':
                if tpo.answer != -1:
                    tpo.is_multi = True
                else:
                    tpo.answer = option_number
    if "lecture" in title:
        tpo.type = "lecture"
    else:
        tpo.type = "conversation"

    return tpo