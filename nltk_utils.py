import nltk
from train import ignore_words
# nltk.download('punkt')
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # Stemmed already tokenized array
    tokenized_sentence = [stem(w) for w in tokenized_sentence if w not in ignore_words]
    bag = np.zeros(len(all_words), dtype=np.float32)

    pass