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

# TOKENIZATION AND ENCODING
def get_byte_pair_encoding(corpus_path, bpe_params):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(**bpe_params)
    files = [corpus_path]
    print(f'files: {files}')
    tokenizer.train(files, trainer)
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

def get_features_and_labels(encodings, sequence_length=4):
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
    
    print(f"Features size: {features.size()}\nLabels size: {labels.size()}")
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

    # Create integer encodings
    encoding = get_integer_encoding(tokens)
    decoding = get_integer_decoding(encoding)
    encodings = [encoding[token] for token in tokens]
    print(f"Vocabulary size: {len(set(encodings))}")
    print(f"Encodings: \n{encodings}")

    # Split
    train_corpus, test_corpus = split(encodings, split=0.9)
    X_train, y_train = get_features_and_labels(train_corpus)
    X_test, y_test = get_features_and_labels(test_corpus)
    print(f"Train dimensions: {X_train.size()}, {y_train.size()}. Test dimensions: {X_test.size()}, {y_test.size()}")

    # Create dataloaders
    batch_size = config['Train_Params']['batch_size']
    train_loader, test_loader = get_dataloaders(X_train, X_test, y_train, y_test, batch_size)
    loaders = (train_loader, test_loader)

    # Save data
    data = (train_loader, test_loader, encoding, decoding, tokenizer)
    save_data(data, goal_path)

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

def save_data(data: str, path: str):
    with open(path, "wb") as file:
        print(f"Now saving data at path {path}...")
        pickle.dump(data, file)
        print("Succesfully saved data.")