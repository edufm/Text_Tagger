from scipy import spatial

class Compare():
    """
    Class that wraps a database to compare texts two tags
    
    args:
        database: the database object with tags the module will work on
                          
    returns:
        Compare object that can compare different tags on teh database
    """
    def __init__(self, database):
        self.database   = database

    def get_similarity(self, tag1, tag_column1, tag2, tag_column2, embeding_method="tf-idf" , dist_method="cos"):
        """ 
        Funcion that compare 2 tags of teh database
        
        args:
            tag1: tag to slice the database with
            tag_column1: column of the database the tag is from
            tag2: second tag to slice the database with
            tag_column2: second column of the database the tag is from
            embeding_method: method that will be used to get the tag, default = "tf-idf"
                             can be ["tf-idf", "cbow", "doc2vec", "lda"]
            dist_method: method taht will be used to measure the distance between tags
            
        returns:
            list of most likely tags for each text
        """
        vectors     = self.database.generate_embedings(embeding_method)
        reindex_df  = self.database.df.reset_index(drop=True)

        if(tag1 not in set(reindex_df[tag_column1]) or
            tag2 not in set(reindex_df[tag_column2])):
            raise ValueError(" ")

        vectors1 = vectors[reindex_df[reindex_df[tag_column1] == tag1].index]
        vectors2 = vectors[reindex_df[reindex_df[tag_column2] == tag2].index]

        
        if dist_method == "cos":
            center_vector1 = vectors1.mean(axis=0)
            center_vector2 = vectors2.mean(axis=0)
            dist = 1 - spatial.distance.cosine(center_vector1, center_vector2)
 
        #spatial.distance.euclidean(a, b)

        #if dist_method == "jaccard":
        #    dist = spatial.distance.cdist(XA=, XB=, metic="jaccard")

        return dist