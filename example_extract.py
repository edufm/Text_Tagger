import text_tagger as tt

# Option to reduce dataframe print size
tt.dataset_manager.pd.options.display.max_colwidth = 20

# User inputs for the database
file = "./datasets_samples/Tweets_USA.csv"
text_column = "Tweet content"
tags_columns = ["Latitude", "Longitude"]

tags_types = {"Lat_Long":("numeric-simple", 10, ["Longitude", "Latitude"])}
filter_flags = {"digits"   : True, "stopwords": True, "text_only": False,
                "simbols"  : True, "punct"    : True, "links"    : True,
                "refs"     : False, "tokenize": True}

languages = ['english']#, 'spanish']
other_stopwords = ["&amp;"]

# Creates the database
database = tt.DataBase(file, text_column, tags_columns)
database.open()

# Preprocess the database
preprocess = tt.Preprocess(tags_types, filter_flags, 
                           languages=languages, other_stopwords=other_stopwords)
preprocess.preprocess(database)

# Creates the index of words for the database
database.create_index()

# Choose the tags that will be used
tag_column = "Lat_Long"
tag = database.df[tag_column].iloc[0]

# Para o extrator
extract = tt.Extract(database)
print("Number of elements in tag", extract.get_size(tag, tag_column))
print()
print("Main Words in tag:")
print(extract.get_words(tag, tag_column, n_words=10))
print()
print("Main Tweets in tag:")
print(*extract.get_text(tag, tag_column), sep="\n")

extract.get_wordcloud(tag, tag_column)

#extract.get_lda(tag, tag_column)