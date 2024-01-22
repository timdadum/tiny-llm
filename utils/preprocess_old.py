import numpy as np
import os
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE


print("Current Working Directory:", os.getcwd())

def get_corpus_chars(corpus: str) -> np.array:
    unique = np.unique(list(corpus))
    return set(unique.flatten())

def tokenize_corpus(corpus: str, vocabulary: set) -> np.array:
    tokens = np.array([])
    i = 0
    while i < len(corpus):
        token = ""
        for j in range(min(64, len(corpus) - i)):
            next_token = token + corpus[i + j]
            if next_token in vocabulary:
                token = next_token  # Extend the token if it's in the vocabulary
            else:
                break  # Stop extending if the new token is not in the vocabulary

        if not token:  # Fallback for characters not in vocabulary
            token = corpus[i]

        tokens = np.append(tokens, token)
        i += len(token)  # Move forward by the length of the token
    
    return tokens

def get_byte_pair_encoding(corpus: str, iters: int=32):
    vocab = get_corpus_chars(corpus)
    
    for iteration in range(iters):
        tokens = tokenize_corpus(corpus, vocab)
        pairs = Counter()

        # Count pairs
        for i in range(len(tokens)-1):
            pair = tokens[i] + tokens[i+1]
            pairs[pair] += 1

        to_merge_pair = pairs.most_common(1)[0][0]
        print(f"Iteration {iteration} most common found pair: {to_merge_pair}. Adding to vocabulary...")
        if pairs[to_merge_pair] == 1:
            break  # Exit if the most common pair only appears once

        # Add the new pair to the vocabulary
        vocab.add(to_merge_pair)

        # Update corpus with the new token
        corpus = corpus.replace(to_merge_pair, to_merge_pair.replace(" ", ""))

    print(f"Final found vocabulary: {vocab}")

    return vocab
        
def read_corpus(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()
    
path = 'tiny-llm/corpuses/shakespeare_short.txt'
if not os.path.exists(path):
    print("Path not found")

corpus = read_corpus(path)
vocabulary = get_corpus_chars(corpus)
subword_vocabulary = get_byte_pair_encoding(corpus, iters=128)
tokens = tokenize_corpus(corpus, vocabulary)