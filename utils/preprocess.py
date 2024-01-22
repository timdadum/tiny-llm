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

def get_byte_pair_encoding(corpus_path, bpe_params):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(**bpe_params)
    files = [corpus_path]
    tokenizer.train(files, trainer)
    return tokenizer
        
def read_corpus(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()
    
def save_data(data: str, path)

path = 'tiny-llm/corpuses/shakespeare.txt'
if not os.path.exists(path):
    print("Path not found")

# Load the JSON file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

### Pytorch data functions and classes
def split(tokens, split=0.8):
    n = len(tokens)
    split_index = round(n * split)
    train = tokens[:split_index]
    test = tokens[split_index:]
    return train, test

def get_features_and_labels(tokens,
                            sequence_length=64):
    for ... in ...:
        features = []
        labels = 'bec'

class Corpus(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

# Extract the BPE parameters
bpe_params = config['BPE_Params']
corpus = read_corpus(path)
tokenizer = get_byte_pair_encoding(path, bpe_params)
data = tokenizer.encode(corpus).tokens

features = []
labels = 

train_features = 
test_features = 

with open('tiny-llm/data/shakespeare_tokenized.pkl', "wb") as file:
    pickle.dump(data, file)
    print("Tokenization succesfull")