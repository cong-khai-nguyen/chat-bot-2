import json
from nltk_utils import tokenize, stem, bag_of_words

with open('intents.json', 'r') as f:
    intents = json.load(f)

print(intents)

# Hold all words in the json data
all_words = []
# Hold all the tag in json data
tags = []
# Hold the tokenized pattern and tag
xy = []
# Words that will be skipped
ignore_words = ['?', '!', '.', ',']
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))