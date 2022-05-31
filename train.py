import json

with open('intents.json', 'r') as f:
    intents = json.load(f)

print(intents)

# Hold all words in the json data
all_words = []
# Hold all the tag in json data
tags = []
# Hold the pattern and tag
xy = []

for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        pass
