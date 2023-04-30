We have made available our three splits - train.json, dev.json and test.json


The files follow the format of a json list, with each element of the list being an example, represented as a dictionary.
The format of each list element is described below: -
1. "passage" -> List of sentences, representing the passage. The blanked sentences [cloze gaps] are represented as <1>, <2>, <3> and so on
2. "candidates" -> List of sentences, each corresponding to a candidate. Note that in our dataset [as you might know already], the candidates are shared, and could also contain distractors
3. "answer_sequence" -> List of 2-tuples of the form (blank number, candidate id). Note that the blank number is 1-indexed , like in the passage (so 2 corresponds to gap <2> you see in the passage) . The candidate id is 0-indexed and maps to the corresponding index in "candidates".  
4. "number_of_blanks" -> The number of blanks
5. "candidate_length" -> The number of candidates
6. "eid": The unique identifier for this example [within this split]
