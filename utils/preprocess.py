import numpy as np
import os
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import json
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

print("Current Working Directory:", os.getcwd())

# TOKENIZATION
def get_byte_pair_encoding(corpus_path, bpe_params):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(**bpe_params)
    files = [corpus_path]
    print(f'files: {files}')
    tokenizer.train(files, trainer)
    return tokenizer
        

# DATALOADER PREPARATION        
def split(tokens, split=0.8):
    """Splits tokenized corpus in train and test set"""
    n = len(tokens)
    split_index = round(n * split)
    train = tokens[:split_index]
    test = tokens[split_index:]
    return train, test

def get_features_and_labels(tokens,
                            sequence_length=64):
    if len(tokens) < sequence_length:
        raise ValueError(f"Tokenized iterable is not long enough. Please increase length to at least {sequence_length + 1} (current: {len(tokens)})")
    
    features, labels = np.array([]), np.array([])
    for i in range(len(tokens) - sequence_length - 1):
        features = tokens[i : i + sequence_length]
        labels = tokens[i + sequence_length + 1]
    return features, labels

def get_dataloaders(X_train, X_test, y_train, y_test, batch_size):
    train_corpus, test_corpus = Corpus(X_train, y_train), Corpus(X_test, y_test)
    train_loader, test_loader = DataLoader(train_corpus, batch_size=batch_size, shuffle=False), DataLoader(test_corpus, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def prepare(config, corpus_path, goal_path):
    # Tokenize
    tokenizer = get_byte_pair_encoding(corpus_path, config['BPE_Params'])
    corpus = read_corpus(corpus_path)
    tokens = tokenizer.encode(corpus).tokens
    print(type(tokens[0]))

    # Split
    train_corpus, test_corpus = split(tokens, split=0.9)
    X_train, y_train = get_features_and_labels(train_corpus)
    X_test, y_test = get_features_and_labels(test_corpus)
    print(f"Train dimensions: {X_train}, {y_train}. Test dimensions: {X_test}, {y_test}")

    # Create dataloaders
    batch_size = config['Train_Params']['batch_size']
    print(f"Loaded batch size at {batch_size}")
    train_loader, test_loader = get_dataloaders(X_train, X_test, y_train, y_test, batch_size)
    loaders = (train_loader, test_loader)

    # Save data
    save_data(loaders, goal_path)

    return loaders

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

def save_data(data: str, path: str):
    with open(path, "wb") as file:
        print(f"Now saving data at path {path}...")
        pickle.dump(data, file)
        print("Succesfully saved data.")