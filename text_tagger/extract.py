import numpy as np

from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

import pyLDAvis.gensim


class Extract():
    """ 
    Class that wraps a dataframe and allow for the extraction of features 
    of each tag sych as main words, texts, wordcloud, and lda interpretations
    
    args:
        database: database object to explore.
        
    returns:
        Explore object with many different funtions that interprete the database    
    """
    def __init__(self, database):

        self.database = database
        self.embedings = {}


    def get_size(self, tag, tag_column):
        """ 
        Funcion to check how many texts there are in a specific tag
        
        args:
            tag: tag to slice the database with
            tag_column: column of the database the tag is from
            
        returns:
            number of texts in dataframe
        """
        return len(self.database.df[self.database.df[tag_column] == tag])


    def get_words(self, tag, tag_column, n_words=5, method="NPMI"):
        """ 
        Funcion that find the most important words in a tag
        
        args:
            tag: tag to slice the database with
            tag_column: column of the database the tag is from
            n_words: number of words in tag, default = 5
            method: method that will be used to gett teh words, default = "NPMI"
                    can be ["P", "PMI", "NPMI", "word2vec"]
            
        returns:
            list of most important words in dataframe
        """
        if method in ["P", "PMI", "NPMI"]:
            kew_words =  self.database.most_important_word(tag, tag_column=tag_column, method=method, n_words=n_words)

        elif method == "word2vec":
            wv = self.database.generate_embedings("word2vec")
            base_words = self.database.most_important_word(tag, tag_column=tag_column, method="NPMI", n_words=n_words)

            kew_words = []
            for word in base_words:

                if word in wv.vocab:
                    new_words = wv.most_similar(word)

                    for new_word, score in new_words:
                        if score > 0.8:
                            kew_words.append(new_word)
                else:
                    kew_words.append(word)
                    
        return kew_words


    def get_text(self, tag, tag_column, n_texts=5, method="tf-idf"):
        """ 
        Funcion that find the most important texts in a tag
        
        args:
            tag: tag to slice the database with
            tag_column: column of the database the tag is from
            n_texts: number of texts in tag, default = 5
            method: method that will be used to get the texts, default = "tf-idf"
                    can be ["tf-idf", "cbow", "doc2vec", "lda"]
            
        returns:
            list of most important texts in dataframe
        """
        if (tag_column not in self.database.df.columns):
            raise ValueError(f"Tag {tag_column} not found in dataset")
        elif tag not in self.database.df[tag_column].to_list():
            raise ValueError(f"Tag {tag} not found in dataset column {tag_column}")

        vectors = self.database.generate_embedings(method)
        reindex_df = self.database.df.reset_index(drop=True)
        vectors = vectors[reindex_df[reindex_df[tag_column] == tag].index]

        center_vector = vectors.mean(axis=0)
        results = cKDTree(np.array(vectors.asformat("array"))).query(center_vector, k=n_texts)

        sents = []
        for n in results[1].flatten():
            sents.append(reindex_df.iloc[n][f"orig_{self.database.text_column}"])

        return sents


    def get_wordcloud(self, tag, tag_column, raw=False, save=False):
        """ 
        Funcion that geneates a wordcloud fr teh tag
        
        args:
            tag: tag to slice the database with
            tag_column: column of the database the tag is from
            raw: if the data used should be the raw data or the preprocessed data, default False
            save: wether to save or not the figure generated in teh ocal folder, default False
            
        returns:
            plots the word cloud of the tag
        """
        if (tag_column not in self.database.df.columns):
            raise ValueError(f"Tag {tag_column} not found in dataset")
        elif tag not in self.database.df[tag_column].to_list():
            raise ValueError(f"Tag {tag} not found in dataset column {tag_column}")

        if raw:
            corpus = self.database.df[self.database.df[tag_column] == tag][f"orig_{self.database.text_column}"]
            corpus = ".".join(corpus.to_list())
        else:
            corpus = self.database.df[self.database.df[tag_column] == tag][self.database.text_column].to_list()
            corpus = list(map(lambda x: " ".join(x), corpus))
            corpus = ".".join(corpus)

        wordcloud = WordCloud(background_color='white',
                              max_words=100,
                              max_font_size=50).generate(str(corpus))

        fig = plt.figure()
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()

        if save:
            fig.savefig(f"WordCloud_{tag}_{tag_column}.png", dpi=900)
            

    def get_similarity(self, tag, tag_column, word1, word2=None, use_pretrained=True):
        """
        Makes a calulation with word vectors to narrow similar words
        args:
            tag: tag to slice the database with
            tag_column: column of the database the tag is from
            word1: first word to use in comparison
            word2: word to get the similarity with the first, if none, 
                   most similar word to the first will be returned, default, none
            use_pretrained; if the previouslyused embeding for this tag should be reused for this calculation

        example:
            word1 = "man", word2="woman" --> 0.8 (high)

            word1 = "man", word2=None --> "Woman"
            
        returns:
            words with highest similarity
        """
        if (tag_column not in self.database.df.columns):
            raise ValueError(f"Tag {tag_column} not found in dataset")
        elif tag not in self.database.df[tag_column].to_list():
            raise ValueError(f"Tag {tag} not found in dataset column {tag_column}")

        if use_pretrained and f"{tag_column}_{tag}" in self.embedings:
            wv = self.embedings[f"{tag_column}_{tag}"]
        else:
            wv = self.database.generate_embedings(method="word2vec", tag=tag, tag_column=tag_column)
            self.embedings[f"{tag_column}_{tag}"] = wv

        if word2 == None:
            return wv.similar_by_word(word1)
        else:
            return wv.similarity(word1, word2)


    def make_word_difference(self, tag, tag_column, positive, negative, use_pretrained=True):
        """
        Makes a calulation with word vectors to narrow similar words
        args:
            tag: tag to slice the database with
            tag_column: column of the database the tag is from
            postive: list of words taht will be added
            negaive: list of words taht will be subtracted
                   most similar word to the first will be returned, default, none
            use_pretrained; if the previouslyused embeding for this tag should be reused for this calculation

        example:
            positive=['woman', 'king'], negative=['man'] --> queen: 0.8965

        returns:
            results of the analogy
        """
        if (tag_column not in self.database.df.columns):
            raise ValueError(f"Tag {tag_column} not found in dataset")
        elif tag not in self.database.df[tag_column].to_list():
            raise ValueError(f"Tag {tag} not found in dataset column {tag_column}")

        if use_pretrained and f"{tag_column}_{tag}" in self.embedings:
            wv = self.embedings[f"{tag_column}_{tag}"]
        else:
            wv = self.database.generate_embedings(method="word2vec", tag=tag, tag_column=tag_column)
            self.embedings[f"{tag_column}_{tag}"] = wv

        return wv.most_similar_cosmul(positive=positive, negative=negative)
        

    def make_analogy(self, tag, tag_column, relation, target, use_pretrained=True):
        """
        Makes an analogy with data from inside the tag
        args:
            tag: tag to slice the database with
            tag_column: column of the database the tag is from
            relation: 2 words with the desired relation
            target: word that the relation will be aplied to
                   most similar word to the first will be returned, default, none
            use_pretrained; if the previouslyused embeding for this tag should be reused for this calculation

        example:
            relation=['man', 'king'], target=['woman'] --> queen: 0.8965

        returns:
            results of the analogy
        """
        if (tag_column not in self.database.df.columns):
            raise ValueError(f"Tag {tag_column} not found in dataset")
        elif tag not in self.database.df[tag_column].to_list():
            raise ValueError(f"Tag {tag} not found in dataset column {tag_column}")

        if use_pretrained and f"{tag_column}_{tag}" in self.embedings:
            wv = self.embedings[f"{tag_column}_{tag}"]
        else:
            wv = self.database.generate_embedings(method="word2vec", tag=tag, tag_column=tag_column)
            self.embedings[f"{tag_column}_{tag}"] = wv
            
        if isinstance(target, list):
            target = target[0]

        positive = [relation[-1], target]
        negative = [relation[0]]

        return wv.most_similar_cosmul(positive=positive, negative=negative)
    
    
    def get_lda(self, tag, tag_column):
        """ 
        Funcion that geneates lda visualization for the tag
        
        args:
            tag: tag to slice the database with
            tag_column: column of the database the tag is from
            
        returns:
            A local server that host the lda visualization
        """
        vectors, model = self.database.generate_embedings(method="lda", tag=tag, tag_column=tag_column, return_model=True)
        
        ldamodel, doc2bow, dictionary = model
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, doc2bow, dictionary, sort_topics=False, mds='mmds')
        pyLDAvis.show(lda_display)