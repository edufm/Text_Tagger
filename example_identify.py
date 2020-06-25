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

# Escolhe a tag que vai ser observada
tag_column = "Lat_Long"
tag = database.df[tag_column].iloc[-1]

# Para o Indentify
identify = tt.Identify(database)
print(identify.identify(["I ran a marathon in los Angeles this week and did not win", 
                         "A ball rows down the stairs while the boy watch it go"], method="cbow"))
    