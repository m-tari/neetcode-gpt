from typing import Any, List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        # Initialize character-level vocabulary
        vocab = list[str](corpus)
        merges = []
        
        # Repeat num_merges times
        for _ in range(num_merges):
            # Count frequency of all adjacent token pairs
            pair_counts = {}
            for i in range(len(vocab) - 1):
                pair = (vocab[i], vocab[i+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            if not pair_counts:
                break
        
            # Find the most frequent pair (break ties lexicographically)
            most_frequent_pair = min(pair_counts.keys(), key=lambda p: (-pair_counts[p], p))
        
            # Merge all non-overlapping occurrences left to right
            merged_vocab = []
            i = 0

            while i < len(vocab):
                if i < len(vocab) - 1 and vocab[i] == most_frequent_pair[0] and vocab[i+1] == most_frequent_pair[1]:
                    merged_vocab.append(most_frequent_pair[0] + most_frequent_pair[1])
                    i += 2
                else:                    
                    merged_vocab.append(vocab[i])
                    i += 1
            
            vocab = merged_vocab
            merges.append(list(most_frequent_pair))
        
        return merges