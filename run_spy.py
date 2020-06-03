import text_tagger as tt

file = "./datasets_samples/Tweets_USA.csv"
text_column = "Tweet content"
tags_columns = ["Latitude", "Longitude"]

tags_types = {"Latitude":("numeric-simple", 200, ["Longitude"])}
filter_flags = {"digits"   : True, "stopwords": True, "text_only": False,
                "simbols"  : True, "punct"    : True, "links"    : True,
                "refs"     : True}

database = tt.DataBase(file, text_column, tags_columns)
database.open()

preprocess = tt.Preprocess(tags_types, filter_flags)
preprocess.preprocess(database)

database.save() 

database.create_index()