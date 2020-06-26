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

# Generates one embeding
database.generate_embedings(method="tf-idf")

# Generate autotags
database.generate_tags()

# Creates the words indexes
database.create_index()

# Choose some randon tags
tag_column = "AutoTag"
tag = database.df[tag_column].iloc[-1]

tag_2, i = tag, -2
while tag_2 == tag:
    tag_2 = database.df[tag_column].iloc[i]
    i -= 1

# Save the data
if False:
    database.export(target="text")
    database.export(target="csv")
    database.save()

# Runs the Extractor
if False:
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

# Runs generate
if False:    
    generate = tt.Generate(database, max_sequence_len=16)
    generate.train(tag, tag_column)
    print(generate.generate(seed_text="Want to"))


# Runs identify
if False:
    identify = tt.Identify(database)
    print(identify.identify(["I ran a marathon in los Angeles this week and did not win", 
                             "A ball rows down the stairs while the boy watch it go"], method="cbow"))
    
# Runs compare
if False:
    compare = tt.Compare(database)
    print(compare.get_similarity(tag, tag_column, tag_2, tag_column))