class Compare():
    def __init__(self, tag1, tag2, database, tag_column, method):
        self.database   = database
        self.tag_column = tag_column
        self.method     = method
        self.tag1       = tag1
        self.tag2       = tag2
        
    def get_embeddings_tags(self, method):
        vectors     = self.database.generate_embedings(method)
        reindex_df  = self.database.df.reset_index(drop=True)

        if(self.tag1 not in set(reindex_df[self.tag_column]) or
            self.tag2 not in set(reindex_df[self.tag_column])):
            raise ValueError(" ")

        vectors1 = vectors[reindex_df[reindex_df[self.tag_column] == self.tag1].index]
        vectors2 = vectors[reindex_df[reindex_df[self.tag_column] == self.tag2].index]

        return vectors1, vectors2

    def get_similarity(self):
        vectors1, vectors2 = self.get_embeddings_tags(self.method)