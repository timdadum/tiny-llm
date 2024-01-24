import utils.preprocess as pre
import json

# Load the JSON file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

# Run preprocessing
path = 'tiny-llm/corpuses/shakespeare_short.txt'
goal_path = 'tiny-llm/data/shakespeare_short.pkl'
loaders = pre.prepare(config, path, goal_path)
