import text_tagger as tt

file = "./datasets_samples/Tweets_USA.csv"
text_columns = "Tweet content"
tags_columns = ["Latitude", "Longitude"]
tags_types = {"Latitude":("numeric-simple", 50, ["Longitude"])}

text_df = tt.preprocess.Preprocess_Text(text_columns).open(file)
tags_df = tt.preprocess.Preprocess_Tags().preprocess(file, tags_columns, tags_types)