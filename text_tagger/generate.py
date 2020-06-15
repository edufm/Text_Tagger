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
    def __init__(self, max_sequence_len):
        self.max_sequence_len = max_sequence_len

    def get_random_seed(self, docs, seed_size=100):
        """
        gera um texto com carracteristicas de sua classe.
        """
        return "Want to"
        #return "Can you recommend"

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

    def sequences_generate(self, docs):
        """
        Args:
            docs:
        """
        sequences = []
        for line in docs:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(2, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                sequences.append(n_gram_sequence)
                        
        sequences = pad_sequences(sequences, maxlen=self.max_sequence_len+1, padding='pre')
        return sequences
    
    def generate_lerolero(self, seed_text, next_words = 20, T = 0.9):
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

    def run(self, database, tag=None, seed_text=None):
        """
        Start the generate component
        Args: 
            tag:  
            repo:
            seed_text:
        """

        # Filtra os dados rederentes a tag escolhida
        docs = database.df[database.df[database.tags_columns[0]]== tag]
        
        # lista de documentos do texto original
        docs = docs[f"orig_{database.text_column}"]
        
        # Se não foi especificado em texto inicial, este e gerado aleatoriamento.
        if(seed_text is None):
            seed_text = self.get_random_seed(docs)
        
        preprocess = Preprocess(tags_types   = None, 
                                filter_flags = {"digits"   : False,
                                                "stopwords": False,
                                                "text_only": False,
                                                "simbols"  : True,
                                                "punct"    : True,
                                                "links"    : True,
                                                "refs"     : False})
        
        docs = list(preprocess.preprocess_text(docs))
        
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(docs)
        self.total_words = len(self.tokenizer.word_index) + 1

        # Gera as sequencias com base nos textos
        sequences = self.sequences_generate(docs)
        
        # Cria o modelo
        self.create_model()
        
        # Treina o modelo
        self.train_model(sequences = sequences)

        # Gera o lero lero
        seed_text = self.generate_lerolero(seed_text)
        print(seed_text)