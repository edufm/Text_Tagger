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

# Creates teh database
database = tt.DataBase(file, text_column, tags_columns)
database.open()

# Preprocess teh databse
preprocess = tt.Preprocess(tags_types, filter_flags, 
                           languages=languages, other_stopwords=other_stopwords)
preprocess.preprocess(database)

# Choose the tags that will be used
tag_column = "Lat_Long"
tag = database.df[tag_column].iloc[-1]

tag_2, i = tag, -2
while tag_2 == tag:
    tag_2 = database.df[tag_column].iloc[i]
    i -= 1

# Performs teh comparison
compare = tt.Compare(database)
print(compare.get_similarity(tag, tag_column, tag_2, tag_column))