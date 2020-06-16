from os import path
import numpy as np

from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

import pyLDAvis.gensim


class Extract():
    def __init__(self, database):

        self.database = database
        self.embedings = {}

    def get_size(self, tag, tag_column):
        
        return len(self.database.df[self.database.df[tag_column] == tag])


    def get_words(self, tag, tag_column, n_words=5, method="NPMI"):

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


    def get_text(self, tag, tag_column, n_sents=5, method="tf-idf"):
        if (tag_column not in self.database.df.columns):
            raise ValueError(f"Tag {tag_column} not found in dataset")
        elif tag not in self.database.df[tag_column].to_list():
            raise ValueError(f"Tag {tag} not found in dataset column {tag_column}")

        vectors = self.database.generate_embedings(method)
        reindex_df = self.database.df.reset_index(drop=True)
        vectors = vectors[reindex_df[reindex_df[tag_column] == tag].index]

        center_vector = vectors.mean(axis=0)
        results = cKDTree(np.array(vectors.asformat("array"))).query(center_vector, k=n_sents)

        sents = []
        for n in results[1].flatten():
            sents.append(reindex_df.iloc[n][f"orig_{self.database.text_column}"])

        return sents


    def get_wordcloud(self, tag, tag_column, raw=False, save=False):
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


        example:
            positive=['woman', 'king'], negative=['man'] --> queen: 0.8965

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

        return wv.most_similar_cosmul(positive=positive, negative=negative)
        

    def make_analogy(self, tag, tag_column, relation, target, use_pretrained=True):
        """
        Makes an analogy with data from inside the tag
        args:


        example:
            relation=['man', 'king'], target=['woman'] --> queen: 0.8965

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
            
        if isinstance(target, list):
            target = target[0]

        positive = [relation[-1], target]
        negative = [relation[0]]

        return wv.most_similar_cosmul(positive=positive, negative=negative)
    
    
    def get_lda(self, tag, tag_column):
        
        vectors, model = self.database.generate_embedings(method="lda", tag=tag, tag_column=tag_column, return_model=True)
        
        ldamodel, doc2bow, dictionary = model
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, doc2bow, dictionary, sort_topics=False, mds='mmds')
        pyLDAvis.show(lda_display)