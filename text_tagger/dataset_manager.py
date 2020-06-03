from collections import defaultdict
import json

import pandas as pd
import numpy as np

class DataBase():
    def __init__(self, path, text_column, tags_columns):
        """
        Classe de dados do modulo

        path: Caminho até o database que se deseja abrir
        text_label  : Coluna do dataframe com os textos
        tags_labels : Colunas do dataframe com as tags numericas
        """
        self.path = path
        self.storage_path = "./storage/" + path.split("/")[-1]

        self.text_column = text_column
        self.tags_columns = tags_columns


    def open(self):
        self.df = pd.read_csv(self.path, index_col=0)


    def load(self):
        self.df = pd.read_csv(self.storage_path, index_col=0)


    def save(self):
        self.df.to_csv(self.storage_path)


    def text(self):
        return self.df[self.text_column]


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
        
        per_tag_index = defaultdict(lambda:defaultdict(int))
        index = defaultdict(lambda:defaultdict(int))
        for doc_id in text_ids:
            
            doc_data = data[doc_id]
            
            for word in doc_data[self.text_column]:
                # Faz o index para o corpus
                index[word][doc_id] +=1

                # Faz o index para a tag
                for tag in self.tags_columns:
                    per_tag_index[doc_data[tag]][word] += 1

        self.word_index = index
        self.word_tag_index = per_tag_index


    def most_important_word(self, tag, tag_column=None, get=3, method="PMI"):
        
        if tag_column == None:
            tag_column = self.tags_columns[0]

        if not tag in self.word_tag_index.keys():
            raise ValueError("Invalid tag {tag}")

        tag_index = self.word_tag_index[tag]
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




