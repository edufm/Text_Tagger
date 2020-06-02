import text_tagger as tt

file = "Tweets_USA.csv"
text_column = "Tweet content"
tags_columns = ["Latitude", "Longitude"]
tags_types = {"Latitude":("numeric-simple", 50, ["Longitude"])}


preprocess = tt.Preprocess(text_label = text_column,
                           tags_labels = tags_columns,
                           tags_types = tags_types)

df = preprocess.preprocess("./datasets_samples/"+file)

tt.dataset_manager.save(df, "./storage/"+file)
