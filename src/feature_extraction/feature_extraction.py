import pandas as pd
import json
from collections import Counter
from functools import reduce
import os


class FeatureExtraction:
    def __init__(self, query_data_path: str, passage_data_path: str):
        self.query_data_path = query_data_path
        self.passage_data_path = passage_data_path
        self.query_data = self.load_query_data()
        self.passage_data = self.load_passage_data()

    def load_query_data(self):
        # load tsv file
        df = pd.read_csv(self.query_data_path, delimiter="\t")
        return df

    def load_passage_data(self):
        with open(self.passage_data_path) as f:
            data = json.load(f)
        return data

    def term_freq(self, doc):
        return Counter(doc)

    def extract_features(self, extract_path):
        n = len(list(self.passage_data.keys()))
        term_freq_dict = dict()

        for i, doc_id in enumerate(self.passage_data.keys()):
            if i % 50 == 0:
                print(f"{(i/n)*100}% complete")
            for pass_id in self.passage_data[doc_id].keys():
                if i % 50 == 0:
                    print(f"{(i/n)*100}% complete")
                passage = self.passage_data[doc_id][pass_id]
                doc_term_freq = self.term_freq(passage)
                term_freq_dict[doc_id] = doc_term_freq
        coll_term_freq = reduce(lambda x, y: x + y, term_freq_dict.values())

        tf_path = os.path.join(extract_path, "doc_term_freq.json")
        coll_tf_path = os.path.join(extract_path, "col_term_freq.json")

        with open(tf_path, "w") as f:
            json.dump(term_freq_dict, f)

        with open(coll_tf_path, "w") as f:
            json.dump(coll_term_freq, f)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("missing argument train, test or dev")
        sys.exit(1)
    elif sys.argv[1] not in ("train", "test", "dev"):
        print("argument to feature_extraction.py must be in ('train', 'test', 'dev')")
        sys.exit(1)

    CUR_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(CUR_DIR, "../../data")
    EXTRACT_DATA_DIR = os.path.join(DATA_DIR, "extracted")
    passage_data_path = os.path.join(EXTRACT_DATA_DIR, "document_passages.json")

    query_data_path = os.path.join(EXTRACT_DATA_DIR, f"{sys.argv[1]}.csv")
    fe = FeatureExtraction(query_data_path, passage_data_path)
    extract_path = os.path.join(DATA_DIR, f"processed/{sys.argv[1]}")
    fe.extract_features(extract_path)
