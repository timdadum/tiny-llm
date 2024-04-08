import utils.preprocess as pre
import json

# Load the JSON file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

# Run preprocessing
run_name = 'debug'
vocab_size = 128
fraction = 0.05

corpus_path = 'tiny-llm/corpuses/shakespeare.txt'
data_goal_path = 'tiny-llm/data/' + run_name + '.pkl'

data = pre.prepare(corpus_path, run_name, data_goal_path, vocab_size, fraction)