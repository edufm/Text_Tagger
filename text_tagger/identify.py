from scipy.spatial import cKDTree
from .preprocess import Preprocess

import numpy as np
import pandas as pd

class Identify():
    """
    Class that wraps a database to identify in whith tag some new texts belong
    
    args:
        database; the database object with tags for the object to refer to
                          
    returns:
        Identify object that can indentify a text according to embedings
    """
    def __init__(self, database):
        self.database = database


    def identify(self, texts, method="tf-idf", n_searches=3):
        """ 
        Funcion that find the most likely tags for a text
        
        args:
            texts: string or list of strings with the text that should be identified
            method: method that will be used to get the tag, default = "tf-idf"
                    can be ["tf-idf", "cbow", "doc2vec", "lda"]
            n_searches: how many tags to return for each text, default = 3
            
        returns:
            list of most likely tags for each text
        """
        if isinstance(texts, pd.Series):
            pass
        elif isinstance(texts, str):
            texts = pd.Series([texts])
        elif isinstance(texts, list):
            texts = pd.Series(texts)
        else:
            raise ValueError("texts must be a string (single text), list of texts or pandas series")

        # Preprocess os dados        
        filter_flags = {"digits"   : True, "stopwords": True, "text_only": False,
                        "simbols"  : True, "punct"    : True, "links"    : True,
                        "refs"     : True, "tokenize":True}
        languages = ['english']#, 'spanish']

        preprocess = Preprocess(None, filter_flags, languages=languages)
        filtered_texts = list(preprocess.preprocess_text(texts))

        # Pega o modelo de embeding e aplica para os texts
        vectors, model = self.database.generate_embedings(method=method, return_model=True)
    
        if method == "tf-idf":
            new_vectors = list(model.transform(filtered_texts).asformat("array"))

        elif method == "cbow":
            new_vectors = []
            for text in filtered_texts:
                vec = np.zeros(model.wv.vector_size)
                for word in text:
                    if word in model.wv.vocab:
                        vec += model.wv.get_vector(word)
                        
                norm = np.linalg.norm(vec)
                if norm > np.finfo(float).eps:
                    vec /= norm
                new_vectors.append(vec)

        elif method == "doc2vec":
            new_vectors = [model.infer_vector(text) for text in filtered_texts]

        elif method == "lda":
            ldamodel, doc2bow, dictionary = model
            
            raw_vecs = [ldamodel.get_document_topics(text) for text in doc2bow]
            new_vectors = []
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
                new_vectors.append(this_vec)

        else:
            raise ValueError(f"Method {method} not implemented yet")

        tags_columns = self.database.tags_columns
        tags = {}
        searcher = cKDTree(vectors.asformat("array"), balanced_tree=False)
        for i, text, vector in zip(range(len(texts)), texts, new_vectors):
            
            results = searcher.query(vector, k=n_searches)
            
            tags[i] = {"Text":text}
            for j, n in enumerate(results[1].flatten()):
                for tag, column in zip(self.database.df.iloc[n][tags_columns].to_list(), tags_columns):
                    tags[i][f"{column}_{j+1}"] = tag

        return pd.DataFrame(tags).T



