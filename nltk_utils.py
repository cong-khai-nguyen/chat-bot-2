import nltk
# nltk.download('punkt')
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # Words that will be skipped
    ignore_words = ['?', '!', '.', ',']
    # Stemmed already tokenized array
    tokenized_sentence = [stem(w) for w in tokenized_sentence if w not in ignore_words]

    # Create a numpy array bag of word
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag