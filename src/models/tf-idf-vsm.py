import pandas as pd
import numpy as np
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class tf_idf_VSM:
    def __init__(self, passages_path):
        self.passages_path = passages_path
        self.load_data()
        self.find_VSM()
    
    def load_data(self):
        self.train = pd.read_csv(self.passages_path)

    def find_VSM(self):
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.train['Passage'])
        print("fitted the reansform :)")


    def get_cosine_sim(self, query):
        query_vec = self.vectorizer.transform([query])
        cos_sim = cosine_similarity(self.X, query_vec)
        return cos_sim
    
    def get_ranked_passages(self, query, max_results = 1):
        temp_df = pd.read_csv(self.passages_path)
        cosine_sim = self.get_cosine_sim(query)
        temp_df['cosine_sim'] = cosine_sim
        temp_df = temp_df.sort_values(by=['cosine_sim'], ascending = False)
        return temp_df.head(max_results)



    

if __name__ == "__main__":
    import os

    CUR_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(CUR_DIR, "../../data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw/WikiPassageQA")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")


    model = tf_idf_VSM(PROCESSED_DATA_DIR+'/passage_df.csv')

    results = model.get_ranked_passages("Apple Microsoft Google Ireland tax")
    print(results)

    passages = results['Passage']
    for p in passages:
        print("\nPassage: ")
        print(p)