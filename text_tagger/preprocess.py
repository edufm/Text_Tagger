from nltk.corpus import stopwords
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import re

class Preprocess():
    """
    Realiza o preprocessamento de texto e tageamento absoluto de tags numericas.
    """

    def __init__(self, tags_types,
                 filter_flags = {"digits"   : True,
                                 "stopwords": True,
                                 "text_only": False,
                                 "simbols"  : True,
                                 "punct"    : True,
                                 "links"    : True,
                                 "refs"     : True}):
        """
        Args:
            filter_flags: Dicionario de configuração para o processamento de texto
            tags_types  : dicionario com nome da coluna: tipo (tupla ou string) 
                        ["absolute", ("numeric-simple", # de divs, [tags]), ("numeric-cluster", # de clusters, [tags])]
                        para tags numeric2d usar (nome da coluna: tipo-numero_unico")
        """
        self.tags_types     = tags_types
        self.filter_flags   = filter_flags


    def filter(self, text):
        # Retira caracteres obrigatórios
        text = re.sub(r"\\n", "", text, flags=re.DOTALL|re.MULTILINE)

        # Retira caracteres opcionais
        if(self.filter_flags["refs"]):
            text = re.sub(r"#\S+", "", text, flags=re.DOTALL|re.MULTILINE)
            text = re.sub(r"@\S+", "", text, flags=re.DOTALL|re.MULTILINE)

        if(self.filter_flags["text_only"]):
            text = re.sub(r"[^a-zA-Z0-9 ]", "", text, flags=re.DOTALL|re.MULTILINE)

        if(self.filter_flags["digits"]):
            text = re.sub(r"\d+", "num", text, flags=re.DOTALL|re.MULTILINE)

        if(self.filter_flags["links"]):
            text = re.sub(r"http\S+", "", text, flags=re.DOTALL|re.MULTILINE)

        if(self.filter_flags["simbols"]):
            text = re.sub(r"[$%^&*/]+", "", text, flags=re.DOTALL|re.MULTILINE)
        
        if(self.filter_flags["punct"]):
            text = re.sub(r"[!?\[\]\{\}\(\);:.,...'\-\+\_\"]", "", text, flags=re.DOTALL|re.MULTILINE)

        # adiciona espaço antes de emojis e aracteres especiais
        text = re.sub(r"([^a-zA-Z0-9 ])", r" \1 ", text, flags=re.DOTALL|re.MULTILINE)
        text = re.sub(r"(num)", r" \1 ", text, flags=re.DOTALL|re.MULTILINE)

        if(self.filter_flags["stopwords"]):
            # tokeniza o texto
            text = text.lower().split(" ")
            
            # Remove stopwords
            to_remove = [""]
            if(self.filter_flags["stopwords"]):
                to_remove += stopwords.words('english')
            
            for word in  to_remove:
                while word in text:
                    text.remove(word)
            
        # Fazer Stemming
        return text


    def preprocess_text(self, text_series):
        return text_series.apply(self.filter) 
    

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
        tags_series = self.tags_series.copy()
        self.tags_series = pd.DataFrame(index=tags_series.index)
        for tag_name, tag_type in self.tags_types.items():
            if tag_type[0].split("-")[0] == "numeric":
                method = tag_type[0].split("-")[1]
                self.tags_series[tag_name] = self.numeric_process(tags_series[tag_type[2]], method=method, n=tag_type[1])

            elif tag_type == "absolute":
                if len(tag_type[-1]) > 1:
                    raise ValueError("Absolute tags cant have more then one subtag")
                else:
                    self.tags_series[tag_name] = tags_series[tag_type[-1]]

            else:
                raise ValueError(f"tag_type {tag_type[0]} does no exist")

    def preprocess(self, database):
        """
        Le o arquivo e processa os dados de texto.
        Args:
            file: nome do arquivo de entrada
        """
        # Abre o arquivo de dados
        print(database.df.columns)

        
        original_text = database.df[database.text_column].copy()
        original_text.name = f"orig_{database.text_column}"
        
        
        self.text_series = database.df[database.text_column]
        self.tags_series = database.df[database.tags_columns]

        self.text_series = self.preprocess_text(self.text_series) 
        self.preprocess_tags()

        df = pd.concat([original_text, self.text_series, self.tags_series], axis=1)
        database.df = df[df[database.text_column].str.len()>4]
        database.tags_columns = list(self.tags_types.keys())