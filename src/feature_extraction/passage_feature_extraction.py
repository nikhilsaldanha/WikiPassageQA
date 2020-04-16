import pandas as pd
import json
from collections import Counter
from functools import reduce
import os


class PassageFeatureExtraction:
    def __init__(self, passage_data_path: str):
        self.passage_data_path = passage_data_path
        self.passage_data = self.load_passage_data()
        self.doc_term_freq = None
        self.col_term_freq = None

    def load_passage_data(self):
        with open(self.passage_data_path) as f:
            data = json.load(f)
        return data

    def term_freq(self, doc):
        return Counter(doc)

    def extract_features(self, extract_path):
        n = len(list(self.passage_data.keys()))
        self.doc_term_freq = dict()
        self.col_term_freq = None

        for i, doc_id in enumerate(self.passage_data.keys()):
            if i % 50 == 0:
                print(f"{(i/n)*100}% complete")

            self.doc_term_freq[doc_id] = {}
            doc_freq = None
            for pass_id in self.passage_data[doc_id].keys():
                passage = self.passage_data[doc_id][pass_id]
                term_frequency = self.term_freq(passage)
                if doc_freq is not None:
                    doc_freq = doc_freq + term_frequency
                else:
                    doc_freq = term_frequency
                self.doc_term_freq[doc_id][pass_id] = term_frequency
            if doc_freq is None:
                continue
            if self.col_term_freq is not None:
                self.col_term_freq = self.col_term_freq + doc_freq
            else:
                self.col_term_freq = doc_freq

        tf_path = os.path.join(extract_path, "doc_term_freq_webap.json")
        coll_tf_path = os.path.join(extract_path, "col_term_freq_webap.json")

        with open(tf_path, "w") as f:
            json.dump(self.doc_term_freq, f)

        with open(coll_tf_path, "w") as f:
            json.dump(self.col_term_freq, f)


if __name__ == "__main__":
    import sys

    CUR_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(CUR_DIR, "../../data")
    EXTRACT_DATA_DIR = os.path.join(DATA_DIR, "extracted")

    passage_data_path = os.path.join(EXTRACT_DATA_DIR, "webap_passages.json")
    fe = PassageFeatureExtraction(passage_data_path)

    extract_path = os.path.join(DATA_DIR, f"processed")
    fe.extract_features(extract_path)
