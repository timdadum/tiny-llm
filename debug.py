from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder

# Define BPE configuration with end_of_word_suffix
bpe_config = {'unk_token': '<UNK>', 'end_of_word_suffix': '</w>'}

def create_tokenizer(vocab_size):
    """
    Creates and saves a tokenizer with BPE configuration.
    """
    tokenizer = Tokenizer(BPE(**bpe_config))
    print(tokenizer.get_config())
    tokenizer.decoder = BPEDecoder(suffix='</w>')

    # List of files to train on
    path = "tiny-llm/corpuses/nature.txt"
    files = [path]
    trainer = BpeTrainer(vocab_size=128, special_tokens=['<UNK>'])

    # Train the tokenizer
    tokenizer.train(files, trainer)

    # Save tokenizer with configuration (entire object)
    tokenizer.save("my_tokenizer")
    print(f"Saved tokenizer at my_tokenizer")

def load_tokenizer():
    """
    Loads the saved tokenizer with consistent configuration.
    """
    # Create tokenizer with same BPE configuration
    tokenizer = Tokenizer(BPE(**bpe_config))
    print(tokenizer.get_config())
    loaded_tokenizer = tokenizer.from_file("my_tokenizer")
    return loaded_tokenizer

if __name__ == "__main__":
    create_tokenizer(vocab_size=1000)  # Replace with actual vocab size
    loaded_tokenizer = load_tokenizer()

    # Test encoding with suffix
    test_sentence = "The lightning struck."
    encoded = loaded_tokenizer.encode(test_sentence)
    print(encoded.input_ids)
