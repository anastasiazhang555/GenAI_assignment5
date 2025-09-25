import random
from collections import defaultdict

class BigramModel:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bigrams = self._build_bigrams()
    
    def _build_bigrams(self):
        bigrams = defaultdict(list)
        for sentence in self.corpus:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                bigrams[words[i]].append(words[i + 1])
        return bigrams
    
    def generate_text(self, start_word, length):
        result = [start_word.lower()]
        current_word = start_word.lower()
        
        for _ in range(length - 1):
            if current_word in self.bigrams:
                next_word = random.choice(self.bigrams[current_word])
                result.append(next_word)
                current_word = next_word
            else:
                break
                
        return " ".join(result)
