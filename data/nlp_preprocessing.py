import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        # 5. Return the padded tensor

        # 1. Build vocabulary
        vocab = sorted(set[str](word for sentence in positive + negative for word in sentence.split()))
        word_to_id = {word: i + 1 for i, word in enumerate[str](vocab)}
        id_to_word = {i + 1: word for i, word in enumerate[str](vocab)}

        # 2. Encode each sentence
        positive_ids = [[word_to_id[word] for word in sentence.split()] for sentence in positive]
        negative_ids = [[word_to_id[word] for word in sentence.split()] for sentence in negative]
        
        # 3. Combine positive + negative
        all_ids = positive_ids + negative_ids
        tensors = [torch.tensor(ids) for ids in all_ids]

        # 4. Pad shorter sequences with 0s
        padded_ids = nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        # 5. Return the padded tensor
        return padded_ids


