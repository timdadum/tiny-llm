import pickle

with open("tiny-llm/data/shakespeare_short.pkl", "rb") as file:
    data = pickle.load(file)

train, test = data

# Iterating over features and labels
for feature, label in train:
    # Do something with feature and label
    print("Feature:", feature)
    print("Label:", label)

