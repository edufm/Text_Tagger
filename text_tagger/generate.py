from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from .preprocess import Preprocess

import numpy as np

class Generate():
    """
    Class that wraps a database to generte new texts accoring to a specific tag
    
    args:
        database; the database object with tags the model will be trained from
        max_sequence_len: the maximum number o words the texts the model 
                          will use to rtrain should have, default = 20
                          
    returns:
        Generate object taht can train a model in a tag and generate new text from it
    """
    def __init__(self, database, max_sequence_len=20):

        self.database = database
        self.max_sequence_len = max_sequence_len
    
    def train(self, tag, tag_column):
        """
        Function taht trains the generate object in the texts of a specific tag
        
        args:
            tag: tag to slice the database with
            tag_column: column of the database the tag is from
        """
        if (tag_column not in self.database.df.columns):
            raise ValueError(f"Tag {tag_column} not found in dataset")
        elif tag not in self.database.df[tag_column].to_list():
            raise ValueError(f"Tag {tag} not found in dataset column {tag_column}")

        # Filter the tags of the chosen tag
        docs = self.database.df[self.database.df[tag_column]== tag]
        
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

        # Generate text sequences for teh model
        sequences = []
        for line in docs:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(2, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                sequences.append(n_gram_sequence)
        
        sequences = pad_sequences(sequences, maxlen=self.max_sequence_len+1, padding='pre')

        # Creates teh model
        self.model = Sequential()
        self.model.add(Embedding(self.total_words, 256, input_length=self.max_sequence_len))
        self.model.add(Bidirectional(LSTM(128)))
        self.model.add(Dense(self.total_words, activation='softmax'))
        optimizer = Adam(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Trains teh model
        sequences = pad_sequences(sequences, maxlen=self.max_sequence_len+1, padding='pre')
        X = sequences[:, :-1]
        y = to_categorical(sequences[:, -1], num_classes=self.total_words)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9)

        self.model.fit(X_train, y_train, epochs=50, batch_size=4096, 
                    validation_data=(X_valid, y_valid))
    

    def generate(self, seed_text, next_words=20, T=0.9):
        """
        Generate a new text with the trained model
        
        Args: 
            seed_text: text the model will try to continue from based on what it learned
            next_words: how many words to generat efoward
            T: temperature, how much the generate will value the higher probabilties for each word
               closer to 1: more realistic and repetitive the model will be, 
               closer to 0: more creative and nonsensical
               
        returns:
            newly generated text
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
            
            seed_text += " " + (index_to_word[predicted] if predicted != 0 else '')

        return seed_text
