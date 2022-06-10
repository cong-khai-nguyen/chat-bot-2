import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

print(intents)

# Hold all words in the json data
all_words = []
# Hold all the tag in json data
tags = []
# Hold the tokenized pattern and tag
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Words that will be skipped
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
print(all_words)
tags = sorted(set(tags))
print(tags)

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    # Getting the location of the tag associated to the pattern in list tags
    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss

# Convert to numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

print(y_train)

class ChatDataset(Dataset):
    def __int__(self):
        self.n_sample = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # dataset[index]
    # return a tuple
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers=2)

model = torch()