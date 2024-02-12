import numpy as np
import os
from collections import Counter
from tokenizers import Tokenizer

from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import json
import pickle as pk

from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset, DataLoader

print("Current Working Directory:", os.getcwd())

# TOKENIZATION AND ENCODING
def get_byte_pair_encoding(corpus_path, bpe_params, save=False, save_path=None):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(**bpe_params)
    files = [corpus_path]
    tokenizer.train(files, trainer)
    
    if save:
        if save_path is None:
            raise ValueError("Please define the tokenizer save path")
        else:
            tokenizer.save(save_path)
    
    return tokenizer

def get_integer_encoding(tokens):
    encoding = dict()
    current = 0
    for token in tokens:
        if token not in encoding:
            encoding[token] = current
            current += 1
    print(f"Resulting encoding: \n {encoding}")
    return encoding     

def get_integer_decoding(encoding):
    return {value: key for key, value in encoding.items()}   

# DATALOADER PREPARATION        
def split(encodings, split=0.8):
    """Splits tokenized corpus in train and test set"""
    n = len(encodings)
    split_index = round(n * split)
    train = encodings[:split_index]
    test = encodings[split_index:]
    return train, test

def get_features_and_labels(encodings, sequence_length=16):
    if len(encodings) < sequence_length:
        raise ValueError(f"Encoded iterable is not long enough. Please increase length to at least {sequence_length + 1} (current: {len(encodings)})")
    
    # Calculate the size of the dataset
    num_samples = len(encodings) - sequence_length - 1

    # Preallocate tensors for features and labels
    features = torch.empty((num_samples, sequence_length), dtype=torch.float16)
    labels = torch.empty(num_samples)

    # Fill the tensors
    for i in range(num_samples):
        features[i] = torch.tensor(encodings[i : i + sequence_length])
        labels[i] = torch.tensor(encodings[i + sequence_length + 1])
    return features, labels

def prepare(config, corpus_path, data_goal_path):
    # Tokenize
    tokenizer = get_byte_pair_encoding(corpus_path, config['BPE_Params'])
    corpus = read_corpus(corpus_path)
    tokens = tokenizer.encode(corpus).tokens

    # Create integer encodings
    encoding = get_integer_encoding(tokens)
    decoding = get_integer_decoding(encoding)
    encodings = [encoding[token] for token in tokens]

    # Split
    train_corpus, test_corpus = split(encodings, split=0.9)
    X_train, y_train = get_features_and_labels(train_corpus)
    X_test, y_test = get_features_and_labels(test_corpus)
    print(f"Train dimensions: {X_train.size()}, {y_train.size()}. Test dimensions: {X_test.size()}, {y_test.size()}")

    # Create dataloaders
    test_set = Corpus(X_test, y_test)
    train_set = Corpus(X_train, y_train)
    
    # Save data
    data = (train_set, test_set, encoding, decoding, tokenizer)
    save_data(data, data_goal_path)

    return data

# Define Pytorch Dataset submodule
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
    train_set, test_set, encoding, decoding, tokenizer = data

    # Save data + encodings
    with open(data_goal_path, "wb") as file:
        print(f"Now saving data at path {data_goal_path}...")
        print(type(data))
        for item in data:
            print(type(item))
        pk.dump((train_set, test_set, encoding, decoding), file)
        print("Succesfully saved data.")