import utils.preprocess as pre
import json

# Load the JSON file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

# Run preprocessing
run_name = 'debug'
fraction = 0.1
unk_threshold = 1e-4

tokenizer_path = config["Files"]["tokenizer"]
corpus_path = config["Files"]["corpus"]
data_path = config["Files"]["data"]

data = pre.prepare(corpus_path, tokenizer_path, data_path, unk_threshold , fraction)