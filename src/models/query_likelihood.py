import json
import pandas as pd
import os
import ast
from ..feature_extraction.passage_feature_extraction import PassageFeatureExtraction
import math
import gc


class QueryLikelihood:
    def __init__(
        self,
        passage_data: dict,
        doc_term_freq: dict,
        col_term_freq: dict,
        smoothing: str = "Dirichlet",
        mu: str = None,
        lamda: str = None,
    ):
        if smoothing not in ("Dirichlet", "JM"):
            raise ValueError("`smoothing` must be in ('Dirichlet', 'JM')")
        self.smoothing = smoothing

        if mu is None:
            self.mu = 0.1
        else:
            self.mu = mu

        if lamda is None:
            self.lamda = 0.1
        else:
            self.lamda = lamda

        self.passage_data = passage_data

        self.doc_term_freq = doc_term_freq
        self.col_term_freq = col_term_freq

        self.doc_size = self.get_document_size()
        self.col_size = self.get_collection_size()
        self.overlap_tf = None
        # input(doc_size)
        # input(col_size)

    def load_dict(self, path):
        with open(path, "r") as f:
            d = json.load(f)
        return d

    def get_document_size(self):
        doc_size = {}
        for doc_id in self.doc_term_freq.keys():
            doc_size[doc_id] = {}
            for pass_id in self.doc_term_freq[doc_id]:
                doc_size[doc_id][pass_id] = sum(
                    list(self.doc_term_freq[doc_id][pass_id].values())
                )
        return doc_size

    def get_collection_size(self):
        return sum(list(self.col_term_freq.values()))

    def get_overlapping_freq(self, vocab):
        overlap_tf = {}
        n = len(vocab)
        for i, term in enumerate(vocab):
            # print(q_term_set)
            if i % 100 == 0:
                print(f"{(i/n)*100}% complete")
            overlap_tf[term] = {}
            for doc_id in self.doc_term_freq.keys():
                overlap_tf[term][doc_id] = {}
                for pass_id in self.doc_term_freq[doc_id]:
                    overlap_tf[term][doc_id][pass_id] = 0

                    if term in self.doc_term_freq[doc_id][pass_id].keys():
                        overlap_tf[term][doc_id][pass_id] = self.doc_term_freq[doc_id][
                            pass_id
                        ][term]
        return overlap_tf

    def fit(self, queries):
        self.queries = queries
        vocab = set([term for query in queries for term in query])
        self.overlap_tf = self.get_overlapping_freq(vocab)

    def compute_score(self, query, doc_id, pass_id):
        score = 0
        for term in query:
            collection_freq = self.col_term_freq.get(term, 0)
            if collection_freq == 0:
                score = 0
                return score
            else:
                score += math.log(
                    self.overlap_tf[term][doc_id][pass_id]
                    + self.mu * collection_freq / self.col_size
                ) / (self.doc_size[doc_id][pass_id] + self.mu)
        return score

    def predict(self):
        """
        [
            [
                {doc_id: 1, pass_id: 2, score: 23},
                ...
            ],
            ...
        ]
        """
        self.score = []
        n = len(queries)
        for i, query in enumerate(queries):
            gc.collect()
            # initialize list
            if i % 50 == 0:
                print(f"{(i/n)*100}% complete")
            self.score.append([])
            for doc_id in self.passage_data:
                for pass_id in self.passage_data[doc_id]:
                    self.score[-1].append(
                        {
                            "doc_id": doc_id,
                            "pass_id": pass_id,
                            "score": self.compute_score(query, doc_id, pass_id),
                        }
                    )

        return self.score


if __name__ == "__main__":
    CUR_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(CUR_DIR, "../../data")
    EXTRACT_DATA_DIR = os.path.join(DATA_DIR, "extracted")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

    passage_data_path = os.path.join(EXTRACT_DATA_DIR, "document_passages.json")
    fe = PassageFeatureExtraction(passage_data_path)
    passage_data = fe.passage_data

    query_path = os.path.join(EXTRACT_DATA_DIR, f"dev.csv")
    query_df = pd.read_csv(query_path)

    feature_extract_path = os.path.join(PROCESSED_DATA_DIR, "dev")
    fe.extract_features(feature_extract_path)
    doc_term_freq = fe.doc_term_freq
    col_term_freq = fe.col_term_freq
    doc_term_freq = json.load(
        open(os.path.join(feature_extract_path, "doc_term_freq.json"), "r")
    )
    col_term_freq = json.load(
        open(os.path.join(feature_extract_path, "col_term_freq.json"), "r")
    )
    ql = QueryLikelihood(passage_data, doc_term_freq, col_term_freq)
    queries = [ast.literal_eval(el) for el in query_df["Question"].tolist()]

    ql.fit(queries)
    score = ql.predict()

