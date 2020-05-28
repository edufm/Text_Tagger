from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re

from sklearn.cluster import KMeans

class Preprocess_Text():
        
    def __init__(self, text_label, filter_flags = {"digits": True, "stopwords": True}):
        self.text_label     = text_label
        self.filter_flags   = filter_flags

    def preprocess(self, database):
        def filter(text):
            if(self.filter_flags["digits"]):
                re.sub(r"[\d]", "", text, flags=re.DOTALL|re.MULTILINE)

            if(self.filter_flags["stopwords"]):
                re.sub('|'.join(stopwords.words('english')), "", text)
                
            return text

        for (key, value) in database[self.text_label].items():
            database["key"] = filter(value)
        
        return database 

    def open(self, file):
        """
        Le o arquivo e processa os dados de texto.
        Args:
            file: nome do arquivo de entrada
        """
        # Abre o arquivo de dados
        database = pd.read_csv(file)
        database = self.preprocess(database)


class Preprocess_Tags():
    def __init__(self):
        pass
    
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
            for column in data.columns():
                tags = data[column].unique().sort()
                tags_ref, i = {}, n
                while i < len(tags):
                    for tag in tags[i-n:i]:
                        tags_ref[tag] = i 
                    i += n

                for tag in tags[i-n:i]:
                    tags_ref[tag] = i 

                new_data[column] = data[column].apply(lambda x: str(tags_ref[x]))
            new_data = new_data.sum(axis=1)

        elif method == "cluster":
            k = KMeans(n_clusters=n)
            k.fit(data.values())

            new_data = pd.Series(k.labels_, index=data.index)

        return new_data

    def preprocess(self, file, tags_columns, tags_types):
        """
        Abre e processa todas as tags em um novo dataframe
        Args:
            file: nome do arquivo de entrada
            tags_types: dicionario com nome da coluna: tipo (tupla ou string) 
                        ["absolute", ("numeric-simple", # de divs, [tags]), ("numeric-cluster", # de clusters, [tags])]
                        para tags numeric2d usar (nome da coluna: tipo-numero_unico")
        """ 
        database = pd.read_csv(file)
        database = database[tags_columns]

        processed = pd.DataFrame(index=database.index)
        done = []
        for tag in tags_columns:
            if tag in tags_types.keys() and tag not in done:
                done.append(tag)
                
                tag_type = tags_types[tag]
                if isinstance(tag_type, tuple):
                    if tag_type[0] == "numeric":
                        method = tag_type[0].split("-")[1]
                        if len(tag_type) == 3:
                            processed[tag] = self.numeric_process(database[[tag]+tag_type[2]], method=method, n=tag_type[1])
                        else:
                            processed[tag] = self.numeric_process(database[[tag]], method=method, n=tag_type[1])
                        for other_tag in tag_type[1]:
                            done.append(other_tag)

                elif tag_type == "absolute":
                    processed[tag] = database[tag]

                else:
                    raise ValueError(f"tag_type {tags_types[tag]} does no exist")

        file_name = file.split("/")[-1]
        processed.to_csv(f"../storage/{file_name}")

        

dt = Preprocess_Text("Tweet content").open("../datasets_samples/Tweets_USA.csv")

preprocess_tags = Preprocess_Tags()
preprocess_tags.preprocess("../datasets_samples/Tweets_USA.csv", ["Latitude", "Longitude"], 
                           {"Latitude":("numeric-simple", 50, ["Longitude"])})