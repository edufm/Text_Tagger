from nltk.corpus import stopwords
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import re

class Preprocess():
    """
    Realiza o preprocessamento de texto e tageamento absoluto de tags numericas.
    """

    def __init__(self, text_label, tags_labels, tags_types,
                 filter_flags = {"digits"   : True,
                                 "stopwords": True,
                                 "text_only": True,
                                 "simbols"  : True,
                                 "punct"    : True,
                                 "links"    : True}):
        """
        Args:
            text_label  : Coluna do dataframe com os textos
            tags_labels : Coluna do dataframe com as tags numericas
            filter_flags: Dicionario de configuração para o precessamento de texto
            tags_types  : dicionario com nome da coluna: tipo (tupla ou string) 
                        ["absolute", ("numeric-simple", # de divs, [tags]), ("numeric-cluster", # de clusters, [tags])]
                        para tags numeric2d usar (nome da coluna: tipo-numero_unico")
        """

        self.text_label     = text_label
        self.tags_labels    = tags_labels
        self.tags_types     = tags_types
        self.filter_flags   = filter_flags

    def preprocess_text(self):
        def filter(text):
            if(self.filter_flags["text_only"]):
                text = re.sub(r"^[a-zA-Z ]*$", "", text, flags=re.DOTALL|re.MULTILINE)

            if(self.filter_flags["digits"]):
                text = re.sub(r"[\d]", "", text, flags=re.DOTALL|re.MULTILINE)

            if(self.filter_flags["links"]):
                text = re.sub("http\S+", "", text, flags=re.DOTALL|re.MULTILINE)
                
            if(self.filter_flags["simbols"]):
                text = re.sub("[@$%^&*#/]+", "", text, flags=re.DOTALL|re.MULTILINE)
            
            if(self.filter_flags["punct"]):
                text = re.sub("[!?.,\[\]\{\};:]+", "", text, flags=re.DOTALL|re.MULTILINE)
    
            text = re.sub(' +', ' ', text, flags=re.DOTALL|re.MULTILINE)
            text = text.lower().split(" ")

            for word in stopwords.words('english'):
                while word in text:
                    text.remove(word)
            
            # Fazer Stemming
            return text

        self.text_series = self.text_series.apply(filter)        


    def preprocess(self, file):
        """
        Le o arquivo e processa os dados de texto.
        Args:
            file: nome do arquivo de entrada
        """
        # Abre o arquivo de dados
        database = pd.read_csv(file, index_col=0)

        self.text_series = database[self.text_label]
        self.tags_series  = database[self.tags_labels]

        self.preprocess_text()
        self.preprocess_tags()

        return pd.concat([self.text_series, self.tags_series], axis=1)

    
    def numeric_process(self, data, method, n):
        """
        Processa as tags se forem numéricas em 2d (por exemplo: latitude, longitude)
        Args:
            data: series do pandas que deve ser dividida
            method: methodo que será usado para a conversão
            n: numeros de divisões ou clusters
        """ 
        if method == 'simple':
            new_data = pd.DataFrame()
            for column in data.columns:
                tags = sorted(data[column].unique())
                tags_ref, i = {}, n
                while i < len(tags):
                    for tag in tags[i-n:i]:
                        tags_ref[tag] = int(i/n)
                    i += n

                for tag in tags[i-n:i]:
                    tags_ref[tag] = int(i/n)

                new_data[column] = data[column].apply(lambda x: str(tags_ref[x]))
            new_data = new_data.sum(axis=1)

        elif method == "cluster":
            k = KMeans(n_clusters=n)
            k.fit(data.values)

            new_data = pd.Series(k.labels_, index=data.index)

        return new_data

    def preprocess_tags(self):
        """
        Abre e processa todas as tags em um novo dataframe
        """  
        done = []
        tags_series = self.tags_series.copy()
        self.tags_series = pd.DataFrame(index=tags_series.index)
        for tag in self.tags_labels:
            if tag in self.tags_types.keys() and tag not in done:
                
                done.append(tag)
                tag_type = self.tags_types[tag]
                if isinstance(tag_type, tuple):
                    if tag_type[0].split("-")[0] == "numeric":
                        method = tag_type[0].split("-")[1]
                        if len(tag_type) == 3:
                            self.tags_series[tag] = self.numeric_process(tags_series[[tag]+tag_type[2]], method=method, n=tag_type[1])
                            for other_tag in tag_type[2]:
                                done.append(other_tag)
                        else:
                            self.tags_series[tag] = self.numeric_process(tags_series[[tag]], method=method, n=tag_type[1])

                elif tag_type == "absolute":
                    pass

                else:
                    raise ValueError(f"tag_type {tags_types[tag]} does no exist")


if __name__ == "__main__":
    preprocess = Preprocess(text_label = "Tweet content",
                            tags_labels = ["Latitude", "Longitude"],
                            tags_types = {"Latitude":("numeric-simple", 50, ["Longitude"])},
                            filter_flags = {"digits"   : True, "stopwords": True, "links" : True})

    df = preprocess.preprocess("../datasets_samples/Tweets_USA.csv")
