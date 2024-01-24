import utils.preprocess as pre
import json

# Load the JSON file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

# Run preprocessing
corpus_path = 'tiny-llm/corpuses/debug.txt'
goal_path = 'tiny-llm/data/debug.pkl'

data = pre.prepare(config, corpus_path, goal_path)