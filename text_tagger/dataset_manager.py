from collections import defaultdict
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
    def __init__(self, path, text_column, tags_columns, low_memory=False):
        """
        Class that holds all data for the module, will be used to feed other classes

        path: path to the dataset that will be used (.CSV file)
        text_label  : dataset column with the texts
        tags_labels : dataset columns with teh tags
        """
        self.path = path
        self.storage_path = "./storage/" + path.split("/")[-1].split(".")[0]

        self.text_column = text_column
        self.tags_columns = tags_columns
        
        self.embedings = {}

        self.low_memory = low_memory


    def open(self):
        """
        Function to open the dataset according to self.path
        """
        self.df = pd.read_csv(self.path, index_col=0)


    def load(self):
        """ 
        Function to load a previoulsy pickled database objet
        """
        with open(self.storage_path + ".pkl", "rb") as file:
            self = pickle.load(file)


    def save(self):
        """
        Function to save the database object as pickle
        """
        self.embedings = {method:(data[0], None) for method, data in self.embedings.items()}
        
        with open(self.storage_path + ".pkl", "wb") as file:
            pickle.dump(self, file)

    
    def export(self, target="text"):
        """
        Function to export dataset data into csv or .txt file
        
        args:
            target: if "csv" will store the dataset as csv with all modifications
                    if "text" will store the text column in a .txt file
                    default = "text"
        """
        if target == "text":
            with open('storage/texts.txt', 'w', encoding='utf8') as file:    
                texts = self.df[self.text_column]            
                for sentence in texts:
                    file.write(" ".join([tok for tok in sentence]) + "\n")

        if target == "csv":
            self.df.to_csv(self.storage_path + ".csv")


    def create_index(self, per_tag=True):
        """
        Function to create a index of words:number of appearences of the corpus documents and each tag
        """
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


    def generate_embedings(self, method="tf-idf", tag=None, tag_column=None, return_model=False):
        """
        Funtion that generates and saves in the object a embeding for the corpus texts
        
        args:
            method: method that will be used to perform the embeding, default = "tf-idf"
                    can be ["tf-idf", "cbow", "doc2vec", "lda"]
            tag: specific tag for the embeing if a specific embeding is 
                 needed, default = None
            tag_column: specific tag_column for the embeing if a specific 
                        embeding is needed, default = None
            return_model: if teh function needs to return only the vectors of the texts 
                          or also the model used to generate such vectors, default = False
                          
        returns:
            embeding vectors for each text in the dataset or tag
            model used to generate embedings if return_model = True
        """
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
            if return_model:
                return self.embedings[method]
            else:
                return self.embedings[method][0]

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
                        if word in model.wv.vocab:
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
        elif "lda" in method:
            if "_" in method:
                NUM_TOPICS = int(method.split("_")[-1])
            else:
                NUM_TOPICS = 20
            
            dictionary = Dictionary(texts)
            doc2bow = [dictionary.doc2bow(text) for text in texts]
            ldamodel = LdaMulticore(doc2bow, num_topics=NUM_TOPICS, 
                                    id2word=dictionary, passes=30)
            
            raw_vecs = [ldamodel.get_document_topics(text) for text in doc2bow]
            
            lda_vecs = []
            for vec in raw_vecs:
                this_vec = []
                curr = 0
                for i in range(ldamodel.num_topics):
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
                
        # Se não estiver fazendo uma versão com tags salva os resultados
        if tag == None and not self.low_memory:
            self.embedings[method] = (vectors, model) 

        if return_model:
            return vectors, model
        else:
            return vectors


    def most_important_word(self, tag, tag_column, n_words=5, method="PMI"):
        """ 
        Funcion that find the most important words in a tag
        
        args:
            tag: tag to slice the database with
            tag_column: column of the database the tag is from
            n_words: number of words in tag, default = 5
            method: method that will be used to gett teh words, default = "NPMI"
                    can be ["P", "PMI", "NPMI"]
            
        returns:
            list of most important words in dataframe
        """        
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


    def generate_tags(self, method="tf-idf", n_tags=10, vectors=None):
        """
        Function that uses a speific method and clustring to genrate new tags 
        for the dataset, those tags will be added as "AutoTag" in teh dataset
        
        args:
            method: method that will be used to perform the clustering, default = "tf-idf"
                    can be ["tf-idf", "cbow", "doc2vec", "lda"]
            n_tags: number of tags to generate
            vectors: user embeding vecors for each text if they wish too generate 
                     tags according to their own embeding system
        """
        if vectors == None:
            vectors = self.generate_embedings(method)        
            
        k = KMeans(n_clusters=n_tags)
        k.fit(vectors)

        new_data = pd.Series(k.labels_, index=self.df.index)

        self.df["AutoTag"] = new_data
        self.tags_columns.append("AutoTag")