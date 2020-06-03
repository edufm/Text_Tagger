from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from collections import defaultdict
import numpy as np

class generate:
    def __init__(self, repo):
        self.repo = repo

    def model_algumacoisa(self, total_words, max_sequence_len):
        self.model = Sequential()
        self.model.add(Embedding(total_words, 256, input_length=max_sequence_len))
        self.model.add(Bidirectional(LSTM(128)))
        self.model.add(Dense(total_words, activation='softmax'))

    def run(self, tag): 
        pass

