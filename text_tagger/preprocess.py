from nltk.corpus import stopwords
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import re

class Preprocess():
    """
    Class that preproccess a database to filter text and convert tags to absolute ones
    
    args:
        tags_types: dictionary mapping tag_name list of configurations: [method, number of clusters, original tags]
                    example: {"Lat_Long":("numeric-simple", 200, ["Longitude", "Latitude"])}
        filter_tags: dictionary with different keys for the text preprocess
        languages: list of languages to be used for teh stopwords
        other_stopwords: list of manual stopwords to be used
        
    returns:
        object capable of filtering a dataframe
    """
    def __init__(self, tags_types,
                 filter_flags = {"digits"   : True,
                                 "stopwords": True,
                                 "text_only": False,
                                 "simbols"  : True,
                                 "punct"    : True,
                                 "links"    : True,
                                 "refs"     : True,
                                 "tokenize" : True},
                 languages=['english'], other_stopwords=[]):
        self.tags_types     = tags_types
        self.filter_flags   = filter_flags
        self.languages = languages
        
        self.other_stopwords = []
        if other_stopwords != []:
            self.other_stopwords = list(np.array([self.filter_text(other) for other in other_stopwords]).flatten())


    def filter_text(self, text):
        """ 
        Funcion to filter a single text according to preprocess object filter_flags
        
        args:
            text: text to filter
            
        returns:
            filtered text
        """
        # Retira caracteres obrigatórios
        text = re.sub(r"\n", " ", text, flags=re.DOTALL|re.MULTILINE)

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
            text = re.sub(r"[$%^&*/@#]+", "", text, flags=re.DOTALL|re.MULTILINE)
        
        if(self.filter_flags["punct"]):
            text = re.sub(r"[!?\[\]\{\}\(\);:.,'\-\+\_\"\…]", "", text, flags=re.DOTALL|re.MULTILINE)

        # adiciona espaço antes de emojis e aracteres especiais
        text = re.sub(r"([^a-zA-Z0-9 ])", r" \1 ", text, flags=re.DOTALL|re.MULTILINE)
        text = re.sub(r"(num)", r" \1 ", text, flags=re.DOTALL|re.MULTILINE)

        # tokeniza o texto
        text = text.lower().split(" ")
        
        # Remove stopwords
        to_remove = [""] + self.other_stopwords
        if(self.filter_flags["stopwords"]):
            for language in self.languages:
                to_remove += stopwords.words(language)
            
        for word in  to_remove:
            while word in text:
                text.remove(word)
        
        # Se não for para tokenizar junta o texto
        if not self.filter_flags["tokenize"]:
            text = " ".join(text)
            
        return text


    def preprocess_text(self, text_series):
        """ 
        Funcion to filter a pd.Series of texts according to preprocess object filter_flags
        
        args:
            text_series: pd.Series of texts to filter
            
        returns:
            pd.Series of filtered texts
        """
        return text_series.apply(self.filter_text) 
    

    def numeric_process(self, data, method, n):
        """
        Process n-d numerical tags into a absolute numerical tag
        
        Args:
            data: pd.Series that must be processed
            method: method that will be used in the conversion simple or cluster
            n: number of divisions or cluesters
            
        returns:
            new tag pd.Series
        """ 
        if method == 'simple':
            new_data = pd.DataFrame()
            for column in data.columns:
                
                # Lists possible tags
                tags = sorted(data[column].unique())
                n = len(tags)//n
                
                # Calculates the new tag for each tag
                tags_ref, i = {}, n
                while i < len(tags):
                    for tag in tags[i-n:i]:
                        tags_ref[tag] = int(i/n)
                    i += n
                    
                for tag in tags[i-n:i]:
                    tags_ref[tag] = int(i/n)
                    
                # Replace the tag in the colunm
                new_data[column] = data[column].apply(lambda x: str(tags_ref[x]))
                
            # Compiles all dimensions in a single mix
            new_data = new_data.sum(axis=1)

        elif method == "cluster":
            k = KMeans(n_clusters=n)
            k.fit(data.values)

            new_data = pd.Series(k.labels_, index=data.index)

        return new_data

    def preprocess_tags(self):
        """
        Function that preprocess all tags columns according to the objet configurations 
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
        Function that recieves a database object and preprocess the data
        
        Args:
            database: database to be preprocessed
        """
        # Abre o arquivo de dados
        original_text = database.df[database.text_column].copy()
        original_text.name = f"orig_{database.text_column}"

        self.text_series = database.df[database.text_column]
        self.tags_series = database.df[database.tags_columns]

        self.text_series = self.preprocess_text(self.text_series) 
        self.preprocess_tags()

        df = pd.concat([original_text, self.text_series, self.tags_series], axis=1)
        database.df = df[df[database.text_column].str.len()>4]
        database.tags_columns = list(self.tags_types.keys())