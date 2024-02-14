import os
from tokenizers import Tokenizer

from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import pickle as pk

import torch
from torch.utils.data import Dataset
import re

# TOKENIZATION AND ENCODING
def create_tokenizer(corpus_path, tokenizer_name):
    # Initialize a tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize the trainer with your desired vocabulary size
    trainer = BpeTrainer(vocab_size=256)

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


# BPE PRE- AND POST-PROCESSING
def bpe_preprocess(text, save=True):
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
        clean_path = text.split(".")[0] + "_clean.txt"
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


# MAIN PREPARATION FUNCTION
def prepare(corpus_path, run_name, data_goal_path):
    # Load and preprocess corpus
    _ = bpe_preprocess(corpus_path)

    # Fit tokenizer and tokenize corpus. Take ids as we'll use these for training.
    tokenizer = create_tokenizer(corpus_path.split(".")[0] + "_clean.txt", run_name)
    corpus = read_corpus(corpus_path.split(".")[0] + "_clean.txt")
    encodings = tokenizer.encode(corpus).ids

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