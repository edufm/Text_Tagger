from collections import defaultdict
import json
import pickle

import pandas as pd
import numpy as np

# Imports para algoritimos de vectorização
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis.gensim

# Import outros módulos
from sklearn.cluster import KMeans
import scipy

# Inicia a classe
class DataBase():
    def __init__(self, path, text_column, tags_columns):
        """
        Classe de dados do modulo

        path: Caminho até o database que se deseja abrir
        text_label  : Coluna do dataframe com os textos
        tags_labels : Colunas do dataframe com as tags numericas
        """
        self.path = path
        self.storage_path = "./storage/" + path.split("/")[-1].split(".")[0]

        self.text_column = text_column
        self.tags_columns = tags_columns
        
        self.doc_embedings = {}
        self.word_embedings = {}


    def open(self):
        self.df = pd.read_csv(self.path, index_col=0)


    def load(self):
        with open(self.storage_path + ".pkl", "rb") as file:
            self = pickle.load(file)


    def save(self):
        with open(self.storage_path + ".pkl", "wb") as file:
            pickle.dump(self, file)

    
    def export(self, target="text"):
        
        if target == "text":
            with open('storage/texts.txt', 'w', encoding='utf8') as file:    
                texts = self.df[self.text_column]            
                for sentence in texts:
                    file.write(" ".join([tok for tok in sentence]) + "\n")

        if target == "csv":
            self.df.to_csv(self.storage_path + ".csv")

    def create_index(self, per_tag=True, save=True):
        '''Indexa os documentos de um corpus.

        Args:
            repo: dicionario que mapeia docid para uma lista de tokens.

        Returns:
            O índice reverso do repositorio: um dicionario que mapeia token para
            lista de docids.
        '''

        data = self.df.T.to_dict()
        text_ids = self.df.index.to_list()
        
        per_tag_index = {tag_column:defaultdict(lambda:defaultdict(int)) for tag_column in self.tags_columns}
        index = defaultdict(lambda:defaultdict(int))
        for doc_id in text_ids:
            
            doc_data = data[doc_id]
            
            for word in doc_data[self.text_column]:
                # Faz o index para o corpus
                index[word][doc_id] +=1

                # Faz o index para a tag
                for tag_column in self.tags_columns:
                    per_tag_index[tag_column][doc_data[tag_column]][word] += 1

        self.word_index = dict(index)
        self.word_tag_index = {key:dict(value) for key, value in per_tag_index.items()}


    def most_important_word(self, tag, tag_column=None, get=3, method="PMI"):
        
        if tag_column == None:
            tag_column = self.tags_columns[0]

        if not tag in self.word_tag_index[tag_column].keys():
            raise ValueError(f"Invalid tag {tag}")

        tag_index = self.word_tag_index[tag_column][tag]
        index_sum = {word:sum(self.word_index[word].values()) for word in self.word_index.keys()}

        total_words_tag = sum(tag_index.values())
        total_words = sum(index_sum.values())

        scores = {}
        for word in tag_index.keys():

            pwordtag = tag_index[word]/total_words_tag
            pword = index_sum[word]/total_words

            if method == "P":
                scores[word] = pwordtag
            elif method == "PMI":
                scores[word] = np.log2(pwordtag/pword)
            elif method == "NPMI":
                scores[word] = np.log2(pwordtag/pword) / -np.log2(pwordtag)

        return sorted(scores, key=scores.get, reverse=True)[:get]


    def generate_embedings(self, method="tf-idf"):

        texts = self.df[self.text_column]

        if method in self.doc_embedings:
            vectors = self.doc_embedings[method]
        
        elif method in self.word_embedings:
            vectors = self.word_embedings[method]

        else:
            if method == "tf-idf":
                model = TfidfVectorizer(min_df=5, 
                                        max_df=0.9, 
                                        max_features=5000, 
                                        sublinear_tf=False, 
                                        analyzer=lambda x: x)
    
                vectors = model.fit_transform(texts)
                self.doc_embedings[method] = vectors
                
            elif method == "word2vec" or  method == "cbow":
                model = gensim.models.Word2Vec(corpus_file='storage/texts.txt',
                                               window=5,
                                               size=200,
                                               seed=42,
                                               iter=100,
                                               workers=4)
                
                vectors = model.wv
                self.word_embedings[method] = vectors
    
                if method == "cbow":
                    vectors = []
                    for text in texts:
                        vec = np.zeros(model.wv.vector_size)
                        for word in text:
                            if word in model:
                                vec += model.wv.get_vector(word)
                                
                        norm = np.linalg.norm(vec)
                        if norm > np.finfo(float).eps:
                            vec /= norm
                        vectors.append(vec)
        
                    vectors = scipy.sparse.csr.csr_matrix(vectors)
                    self.doc_embedings[method] = vectors
                
            elif method == "doc2vec":
    
                doc2vec = gensim.models.Doc2Vec(
                                corpus_file='storage/texts.txt',
                                vector_size=200,
                                window=5,
                                min_count=5,
                                workers=12,
                                epochs=100)
    
                vectors = scipy.sparse.csr.csr_matrix(doc2vec.docvecs.vectors_docs)
                self.doc_embedings[method] = vectors
                
            else:
                raise ValueError(f"Method {method} is not recognized")
            
        return vectors


    def generate_tags(self, method="tf-idf", n_tags=10, vectors=None):

        if vectors == None:
            vectors = self.generate_embedings(method)        
            
        k = KMeans(n_clusters=n_tags)
        k.fit(vectors)

        new_data = pd.Series(k.labels_, index=self.df.index)

        self.df["AutoTag"] = new_data
        self.tags_columns.append("AutoTag")