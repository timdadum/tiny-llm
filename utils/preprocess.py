import pickle as pk
import torch
from torch.utils.data import Dataset
import re
from utils.tokenization import * 

# TOKENIZATION AND ENCODING
"""
unk_token = "<UNK>"
spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]

def create_tokenizer(tokenizer, trainer, corpus_path, tokenizer_path, vocab_size):
    # Initialize a tokenizer with BPE model
    bpe_config = {'unk_token': '<UNK>', 'end_of_word_suffix': '</w>'}
    tokenizer = Tokenizer(BPE(**bpe_config))
    tokenizer.pre_tokenizer = Whitespace()
    # Use a BPE decoder to reverse the tokenization
    tokenizer.decoder = BPEDecoder(suffix='</w>')

    # Initialize the trainer with your desired vocabulary size
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=['<UNK>', '<PAD>', '<CLS>', '<SEP>'])

    # List of files to train on
    print(f"Corpus path: {corpus_path}")
    files = [corpus_path]

    # Train the tokenizer
    tokenizer.train(files, trainer)

    # Saving the tokenizer
    tokenizer.model_kwargs = bpe_config
    tokenizer.save(tokenizer_path)
    print(f"Succesfully saved tokenizer at {tokenizer_path}")
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

def get_features_and_labels(encodings, t=32, vocab_size=None):
    if len(encodings) < t:
        raise ValueError(f"Encoded iterable is not long enough. Please increase length to at least {t + 1} (current: {len(encodings)})")
    
    # Calculate the size of the dataset. Omit last (possibly incomplete) sample
    num_samples = len(encodings) - t - 1

    # Preallocate tensors for features and labels
    features = torch.empty((num_samples, t), dtype=torch.float16)
    labels = torch.empty((num_samples, t), dtype=torch.long)

    # Fill the tensors
    for i in range(num_samples):
        features[i] = torch.tensor(encodings[i : i + t])
        labels[i] = torch.tensor(encodings[i + 1 : i + 1 + t], dtype=torch.long)

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

def take_corpus_fraction(corpus, fraction):
    # Take fraction of corpus
    n = round(fraction * len(corpus))
    print(f"Taking {n} characters out of total {len(corpus)}: {fraction*100}%")
    corpus_fraction = corpus[:n]
    return corpus_fraction

def create_sequences(config, tokens):
    train_corpus, test_corpus = split(tokens, split=0.8)
    X_train, y_train = get_features_and_labels(train_corpus, t=config['Data_Params']['sequence_length'], vocab_size=config['BPE_Params']['vocab_size'])
    X_test, y_test = get_features_and_labels(test_corpus, t=config['Data_Params']['sequence_length'], vocab_size=config['BPE_Params']['vocab_size'])
    print(f"Train dimensions: {X_train.size()}, {y_train.size()}. Test dimensions: {X_test.size()}, {y_test.size()}")
    return X_train, y_train, X_test, y_test

# MAIN PREPARATION FUNCTION
def prepare(config, fraction=1.0):
    # Load and preprocess corpus
    corpus = read_corpus(config['Files']['corpus'])
    if fraction != 1.0:
        corpus = take_corpus_fraction(corpus, fraction)
    
    # Train a tokenizer (automatically saves), tokenize corpus
    tokenizer = train_tokenizer(config)
    
    print(tokenizer.encode(corpus[:100]).tokens)
    tokens = tokenizer.encode(corpus).ids

    # Split training sequences
    X_train, y_train, X_test, y_test = create_sequences(config, tokens)

    # Create dataloaders
    test_set = Corpus(X_test, y_test)
    train_set = Corpus(X_train, y_train)
    
    # Save data
    data = (train_set, test_set)
    save_data(data, config['Files']['data'])

    return data
