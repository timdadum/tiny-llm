import pickle
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer = tokenizer.from_file("tiny-llm/tokenizers/debug.json")