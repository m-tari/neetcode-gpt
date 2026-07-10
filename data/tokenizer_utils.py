from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        return [self.tokenize_number(str(number), vocab) for number in numbers]
    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self.tokenize_number(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        return round(self.count_tokens(text, vocab) / len(text.split()), 4)

    def tokenize_number(self, number: str, vocab: Dict[str, int]) -> List[str]:
        # Tokenize a single number using greedy left-to-right longest match.
        # Return a list of tokens showing how the number gets split.
        tokens = []
        i = 0
        while i < len(number):
            for j in range(len(number) - i, 0, -1): 
                token = number[i:i+j]
                if token in vocab:
                    tokens.append(token)
                    i += j
                    break
            else:
                tokens.append(number[i])
                i += 1
        return tokens