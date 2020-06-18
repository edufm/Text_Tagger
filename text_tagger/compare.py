from scipy import spatial

class Compare():
    def __init__(self, database):
        self.database   = database
 
    def get_embeddings_tags(self, tag_column1, tag_column2, tag1, tag2, method):
        vectors     = self.database.generate_embedings(method)
        reindex_df  = self.database.df.reset_index(drop=True)

        if(tag1 not in set(reindex_df[tag_column1]) or
            tag2 not in set(reindex_df[tag_column2])):
            raise ValueError(" ")

        vectors1 = vectors[reindex_df[reindex_df[tag_column1] == tag1].index]
        vectors2 = vectors[reindex_df[reindex_df[tag_column2] == tag2].index]

        return vectors1, vectors2

    def get_similarity(self, tag_column1, tag_column2, tag1, tag2, embeding_method="tf-idf" , dist_method="cos"):
        vectors1, vectors2 = self.get_embeddings_tags(tag_column1, tag_column2, tag1, tag2, embeding_method )
        
        if dist_method == "cos":
            center_vector1 = vectors1.mean(axis=0)
            center_vector2 = vectors2.mean(axis=0)
            dist = 1 - spatial.distance.cosine(center_vector1, center_vector2)
 
        #spatial.distance.euclidean(a, b)

        #if dist_method == "jaccard":
        #    dist = spatial.distance.cdist(XA=, XB=, metic="jaccard")

        return dist