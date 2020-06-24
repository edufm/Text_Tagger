from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from .preprocess import Preprocess
from collections import defaultdict

import numpy as np

class Generate():
    def __init__(self, database, max_sequence_len=20):

        self.database = database
        self.max_sequence_len = max_sequence_len


    def create_model(self):
        """
        Args: 
            total_words:
            max_sequence_len:
        """
        self.model = Sequential()
        self.model.add(Embedding(self.total_words, 256, input_length=self.max_sequence_len))
        self.model.add(Bidirectional(LSTM(128)))
        self.model.add(Dense(self.total_words, activation='softmax'))
        optimizer = Adam(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    def train_model(self, sequences):
        """
        Args: 
            sequences:
        """
        sequences = pad_sequences(sequences, maxlen=self.max_sequence_len+1, padding='pre')
        X = sequences[:, :-1]
        y = to_categorical(sequences[:, -1], num_classes=self.total_words)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9)

        self.model.fit(X_train, y_train, epochs=50, batch_size=4096, 
                    validation_data=(X_valid, y_valid))

    
    def train(self, tag, tag_column):
        """
        Start the generate component
        Args: 
            tag:
            repo:
        """
        if (tag_column not in self.database.df.columns):
            raise ValueError(f"Tag {tag_column} not found in dataset")
        elif tag not in self.database.df[tag_column].to_list():
            raise ValueError(f"Tag {tag} not found in dataset column {tag_column}")

        # Filtra os dados rederentes a tag escolhida
        docs = self.database.df[self.database.df[tag_column]== tag]
        
        # lista de documentos do texto original
        docs = docs[f"orig_{self.database.text_column}"]
    
        preprocess = Preprocess(tags_types   = None, 
                                filter_flags = {"digits"   : False,
                                                "stopwords": False,
                                                "text_only": False,
                                                "simbols"  : True,
                                                "punct"    : True,
                                                "links"    : True,
                                                "refs"     : False,
                                                "tokenize" : False})
        
        docs = list(preprocess.preprocess_text(docs))
        
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(docs)
        self.total_words = len(self.tokenizer.word_index) + 1

        # Gera as sequencias com base nos textos
        sequences = []
        for line in docs:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(2, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                sequences.append(n_gram_sequence)
        
        sequences = pad_sequences(sequences, maxlen=self.max_sequence_len+1, padding='pre')

        # Cria o modelo
        self.create_model()

        # Treina o modelo
        self.train_model(sequences = sequences)
    

    def generate(self, seed_text, next_words=20, T=0.9):
        """
        Args: 
            seed_text:
            next_words:
            T:
        """

        index_to_word = {index: word for word, index in self.tokenizer.word_index.items()}

        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len, padding='pre')

            probas = self.model.predict(token_list, verbose=0)
            probas = np.array(probas[0][1:])
            probas = probas ** (1.0 / T)
            probas /= np.sum(probas)
            predicted = np.random.choice(range(1,self.total_words), p=probas)
            # predicted = model.predict_classes(token_list, verbose=0)[0]
            
            seed_text += " " + (index_to_word[predicted] if predicted != 0 else '')

        return seed_text
