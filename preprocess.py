import utils.preprocess as pre
import json

# Load the JSON file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

# Run preprocessing
run_name = 'shakespeare'
fraction = 0.1

encoding_path = f'tiny-llm/encodings/' + run_name + '.json'
corpus_path = f'tiny-llm/corpuses/' + run_name + '.txt'
data_goal_path = f'tiny-llm/data/' + run_name + '.pkl'

data = pre.prepare(corpus_path, encoding_path, data_goal_path, fraction)