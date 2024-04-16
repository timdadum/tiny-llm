from model_classes.gpt import GPTTokenizer
from transformers import AutoTokenizer
from tokenizers.decoders import BPEDecoder

from tokenizers import Tokenizer

from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

def prepare_tokenizer_trainer(config, unk_token, spl_tokens, alg='BPE'):
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    trainer = BpeTrainer(vocab_size=config['BPE_Params']['vocab_size'], special_tokens=spl_tokens)
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer

def train_tokenizer(config, unk_token='<UNK>', spl_tokens=["<UNK>", "<SEP>", "<MASK>", "<CLS>"], alg='BPE'):
    tokenizer, trainer = prepare_tokenizer_trainer(config, unk_token, spl_tokens)
    tokenizer.train(files=[config['Files']['corpus']], trainer=trainer)
    
    # Save and load the file - checks whether saving works.
    tokenizer.save(config['Files']["tokenizer"])
    tokenizer = Tokenizer.from_file(config['Files']['tokenizer'])
    return tokenizer

def load_tokenizer(config):
    return Tokenizer.from_file(config['Files']['tokenizer'])
