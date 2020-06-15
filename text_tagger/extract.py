
from os import path
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

class Extract():
    def __init__(self, tag, tag_column, method="tf-idf"):

        self.tag = tag
        self.tag_column = tag_column
        self.method = method

    def get_words(self, database, n_words):

        if self.method in ["P", "PMI", "NPMI"]:
            kew_words =  database.most_important_word(self.tag, tag_column=self.tag_column, method=self.method)

        elif self.method == "word2vec":
        
            wv = database.generate_embedings("word2vec")
            base_words = database.most_important_word(self.tag, tag_column=self.tag_column, method="NPMI")

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


    def get_text(self, database, n_sents):

        pass


    def word_cloud(self, database, raw=False, save=False):

        if raw:
            corpus = database.df[database.df[self.tag_column] == self.tag][f"orig_{database.text_column}"]
            corpus = ".".join(corpus.to_list())
        else:
            corpus = database.df[database.df[self.tag_column] == self.tag][database.text_column].to_list()
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
            fig.savefig(f"WordCloud_{self.tag}_{self.tag_column}.png", dpi=900)