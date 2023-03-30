from utils import *
import os
from typing import List
import matplotlib.pyplot as plt

files = os.listdir("./TOEFL-QA/data")

tpos: List[Tpo] = []
for file in files:
    path = os.path.join("./TOEFL-QA/data", file)
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    tpos.append(parse_text(content, file))

length = []
for i in tpos:
    length.append(len(i.sentence))
print(length)
plt.hist(length, bins=40)
plt.show()


