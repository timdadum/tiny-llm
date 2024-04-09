import os
from tokenizers import Tokenizer

from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import pickle as pk
import json

import torch
from torch.utils.data import Dataset
import numpy as np
import re

# TOKENIZATION AND ENCODING
"""
def create_tokenizer(corpus_path, tokenizer_name, vocab_size):
    # Initialize a tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize the trainer with your desired vocabulary size
    trainer = BpeTrainer(vocab_size=vocab_size)

    # List of files to train on
    files = [corpus_path]

    # Train the tokenizer
    tokenizer.train(files, trainer)

    # Saving the tokenizer
    save_path = f'tiny-llm/tokenizers/{tokenizer_name}.json'
    print(f"Save path: {save_path}")
    tokenizer.save(save_path)
    print(f"Succesfully saved tokenizer at {save_path}")
    return tokenizer
"""
    
# DATALOADER PREPARATION        
def split(encodings, split=0.8):
    """Splits tokenized corpus in train and test set"""
    n = len(encodings)
    split_index = round(n * split)
    train = encodings[:split_index]
    test = encodings[split_index:]
    return train, test

def get_features_and_labels(encodings, t=32):
    if len(encodings) < t:
        raise ValueError(f"Encoded iterable is not long enough. Please increase length to at least {t + 1} (current: {len(encodings)})")
    
    # Calculate the size of the dataset. Omit last (possibly incomplete) sample
    num_samples = len(encodings) - t - 1

    # Preallocate tensors for features and labels
    features = torch.empty((num_samples, t), dtype=torch.float16)
    labels = torch.empty(num_samples)

    # Fill the tensors
    for i in range(num_samples):
        features[i] = torch.tensor(encodings[i : i + t])
        labels[i] = torch.tensor(encodings[i + t + 1])
    return features, labels


# Define Pytorch Dataset subclass
class Corpus(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    

# DATA I/O
def read_corpus(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def save_data(data, data_goal_path: str):
    with open(data_goal_path, "wb") as file:
        print(f"Now saving data at path {data_goal_path}...")
        pk.dump(data, file)
        print("Succesfully saved data.")

def save_encoding(data, path: str):
    with open(path, 'w') as file:
        json.dump(data, file)

# WORD-ENCODING PRE-PROCESSING
def word_encoding(text, thresh=1e-5):
    """Encodes a text using an integer encoding and word-level tokenization"""
    # Split text
    tokens = np.array(text.split())

    # Mask rare words using '[UNK]'
    unique, counts = np.unique(tokens, return_counts=True)
    threshold = np.round(thresh*len(tokens))
    rare_tokens = unique[counts < threshold]
    print(f"Threshold is {threshold} occurences. Percentage of rare tokens: {(len(rare_tokens)/len(unique))*100}%")
    tokens = np.where(np.isin(tokens, rare_tokens), '<UNK>', tokens)
        
    # Apply integer encoding
    unique = np.unique(tokens)
    print(f"This threshold would lead to a vocabulary of {len(unique)}")
    mapping = {token: i for i, token in enumerate(unique)}
    print(mapping)
    vectorized_map = np.vectorize(lambda x: mapping.get(x,x))
    y = vectorized_map(tokens)
    return (y, mapping)

'''
# BPE PRE- AND POST-PROCESSING
def bpe_preprocess(text, save=True, corpus_path=None):
    """Adds special tokens for the model to learn. Overview of special tokens:
    _ : Start of word
    """
    special_token_mapping = {' ': ' _'}

    # First lowercase
    text = text.lower()

    # Then add special characters
    for token, new_token in special_token_mapping.items():
        text = text.replace(token, new_token)

    if save:
        if corpus_path is None:
            raise ValueError("Please set save path for .bpe_process(save_path=...)")
        else:
            clean_path = corpus_path.split(".")[0] + "_clean.txt"
            with open(clean_path, "w", encoding='utf-8') as f:
                f.write(text)

    return text

def bpe_postprocess(output):
    """Removes special tokens and makes model output human readable again conform the conventions in preprocessing"""
    post_process_mapping = {' ': '',
                            '_': ' '}

    # Interpret and remove special characters
    for token, new_token in post_process_mapping.items():
        output = output.replace(token, new_token)

    # Re-capitalize corpus
    def capitalize_after_period(text):
        return re.sub(r"(?<=\.)(\w)", lambda match: match.group(1).upper(), text)
    output = capitalize_after_period(output)
    return output
'''

# MAIN PREPARATION FUNCTION
def prepare(corpus_path, encoding_path, data_goal_path, fraction=1.0):
    # Load and preprocess corpus
    corpus = read_corpus(corpus_path)
    encodings, mapping = word_encoding(corpus)
    save_encoding(mapping, encoding_path)

    # Take fraction of corpus
    n = round(fraction * len(corpus))
    print(f"Taking {n} characters out of total {len(encodings)}: {fraction*100}%")
    encodings = encodings[:n]

    print(encodings)
    
    # Split training sequences
    train_corpus, test_corpus = split(encodings, split=0.8)
    X_train, y_train = get_features_and_labels(train_corpus)
    X_test, y_test = get_features_and_labels(test_corpus)
    print(f"Train dimensions: {X_train.size()}, {y_train.size()}. Test dimensions: {X_test.size()}, {y_test.size()}")

    # Create dataloaders
    test_set = Corpus(X_test, y_test)
    train_set = Corpus(X_train, y_train)
    
    # Save data
    data = (train_set, test_set)
    save_data(data, data_goal_path)

    return data
