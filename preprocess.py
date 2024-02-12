import utils.preprocess as pre
import json

# Load the JSON file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

# Run preprocessing
run_name = 'shakespeare'

corpus_path = 'tiny-llm/corpuses/' + run_name + '.txt'
data_goal_path = 'tiny-llm/data/' + run_name + '.pkl'
tokenizer_goal_path = 'tiny-llm/tokenizers/' + run_name + '.json'

data = pre.prepare(config, corpus_path, data_goal_path, tokenizer_goal_path)