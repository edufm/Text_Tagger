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
        
        self.embedings = {}


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


    def most_important_word(self, tag, tag_column=None, n_words=5, method="PMI"):

        if (tag_column not in self.df.columns):
            raise ValueError(f"Tag {tag_column} not found in dataset")
        elif tag not in self.df[tag_column].to_list():
            raise ValueError(f"Tag {tag} not found in dataset column {tag_column}")

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

        return sorted(scores, key=scores.get, reverse=True)[:n_words]


    def generate_embedings(self, method="tf-idf", tag=None, tag_column=None, return_model=False):
        # Coleta os dados dos embedings e salva em um arquivo par ao multiprocess
        if tag != None and tag_column != None:
            if (tag_column not in self.df.columns):
                raise ValueError(f"Tag {tag_column} not found in dataset")
            elif tag not in self.df[tag_column].to_list():
                raise ValueError(f"Tag {tag} not found in dataset column {tag_column}")
            texts = self.df[self.df[tag_column]==tag][self.text_column]
        else:
            texts = self.df[self.text_column]

        with open('storage/texts.txt', 'w', encoding='utf8') as file:               
            for sentence in texts:
                file.write(" ".join([tok for tok in sentence]) + "\n")

        # Verifica se usuario cometeu um erro no imput das tags
        if tag != None and tag_column == None:
            raise ValueError("if passing tag must pass tag_column as well")
        
        if tag_column != None and tag == None:
            raise ValueError("if passing tag_column must pass tag as well")

        # Verifica se o vetor ja foi gerado e se o alvo é o corpus inteiro
        if method in self.embedings and tag == None:
            vectors = self.embedings[method]

        # Se os vetores não tiverem sdo gerados faz o embedding
        else:
            # Realiza o tf-idf
            if method == "tf-idf":
                model = TfidfVectorizer(min_df=5, 
                                        max_df=0.9, 
                                        max_features=5000, 
                                        sublinear_tf=False, 
                                        analyzer=lambda x: x)
    
                vectors = model.fit_transform(texts)
            
            # Realiza o Word2Vec
            elif method == "word2vec" or  method == "cbow":
                model = gensim.models.Word2Vec(corpus_file='storage/texts.txt',
                                               window=5, size=200, min_count=5,
                                               iter=100, workers=4)
                
                vectors = model.wv
                if tag == None:
                    self.embedings["word2vec"] = vectors 

                # Realiza o cbow
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
            
            # Realiza o Doc2Vec
            elif method == "doc2vec":
    
                model = gensim.models.Doc2Vec(
                                corpus_file='storage/texts.txt',
                                vector_size=200, window=5,
                                min_count=5, workers=12,
                                epochs=100)
    
                vectors = scipy.sparse.csr.csr_matrix(model.docvecs.vectors_docs)
              
            # Realiza a LDA
            elif method == "lda":
                NUM_TOPICS = 20
                
                dictionary = Dictionary(texts)
                doc2bow = [dictionary.doc2bow(text) for text in texts]
                ldamodel = LdaMulticore(doc2bow, num_topics=20, id2word=dictionary, passes=30)
                
                raw_vecs = [ldamodel.get_document_topics(text) for text in doc2bow]
                
                lda_vecs = []
                for vec in raw_vecs:
                    this_vec = []
                    curr = 0
                    for i in range(NUM_TOPICS):
                        if (i == vec[curr][0]):
                            this_vec.append(vec[curr][1])
                            curr+=1
                            if curr == len(vec):
                                curr = -1
                        else:
                            this_vec.append(0)
                    lda_vecs.append(this_vec)
                    
                vectors = scipy.sparse.csr.csr_matrix(lda_vecs)
                model = [ldamodel, doc2bow, dictionary]
                
            else:
                raise ValueError(f"Method {method} is not recognized")
                
        if tag == None:
            self.embedings[method] = vectors 

        if return_model:
            return vectors, model
        else:
            return vectors


    def generate_tags(self, method="tf-idf", n_tags=10, vectors=None):

        if vectors == None:
            vectors = self.generate_embedings(method)        
            
        k = KMeans(n_clusters=n_tags)
        k.fit(vectors)

        new_data = pd.Series(k.labels_, index=self.df.index)

        self.df["AutoTag"] = new_data
        self.tags_columns.append("AutoTag")