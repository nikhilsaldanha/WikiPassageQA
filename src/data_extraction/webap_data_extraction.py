import xmltodict
import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd


class WebAPDataExtraction:
    def __init__(
        self, query_data_path, passage_data_path, relevant=("PERFECT", "EXCEL", "GOOD")
    ):
        self.query_data = self.load_query_data(query_data_path)
        self.passage_data = self.load_passage_data(passage_data_path)
        self.relevance_ids = {"PERFECT": 1, "EXCEL": 2, "GOOD": 3, "FAIR": 4, "NONE": 5}
        self.relevant = relevant

    def load_query_data(self, data_path: str):
        with open(data_path, "r") as f:
            query_data = json.load(f)

        query_data = pd.DataFrame.from_records(query_data["queries"])
        return query_data

    def load_passage_data(self, data_path: str):
        with open(data_path, "r", encoding="utf8") as f:
            passage_data_xml = f.read()

        passage_data_dict = xmltodict.parse(passage_data_xml)

        return passage_data_dict

    def preprocess(self, passage: str):
        punct_dict = str.maketrans("", "", string.punctuation)
        wordnet_lemmatizer = WordNetLemmatizer()
        stop_word_list = stopwords.words("english")

        passage = passage.lower()
        passage = passage.translate(punct_dict)
        passage = nltk.word_tokenize(passage)
        # remove numbers
        passage = [word for word in passage if not word.replace(".", "", 1).isdigit()]
        passage = [word for word in passage if word not in stop_word_list]
        passage = [wordnet_lemmatizer.lemmatize(word) for word in passage]

        return passage

    def extract_data(
        self,
        query_extract_path: str,
        passages_extract_path: str,
        preprocess: bool = False,
    ):
        docs = self.passage_data["ROOT"]["DOC"]
        passages = {}

        query_dfs = []
        for di, doc in enumerate(docs):
            query_id = doc["TARGET_QID"]
            doc_id = doc["ORIGINAL_DOCNO"]
            texts = doc["TEXT"]
            passages[doc_id] = {}
            queries = []
            for relevance in texts:
                paras = texts[relevance]
                if type(paras) is list:
                    pass
                elif len(paras) == 1 and type(paras):
                    paras = [paras]
                for i, para in enumerate(paras):
                    passage_id = f"{self.relevance_ids[relevance]}{i}"
                    query_text = self.query_data[self.query_data["number"] == query_id][
                        "text"
                    ]
                    if len(query_text) > 0:
                        if relevance in self.relevant:
                            queries.append(
                                {
                                    "DocumentID": doc_id,
                                    "QID": query_id,
                                    "Question": self.preprocess(query_text.values[0]),
                                    "RelevantPassages": passage_id,
                                }
                            )
                        passage = " ".join(para["SENTENCE"])
                        # pre-processing
                        if preprocess:
                            passage = self.preprocess(passage)
                        passages[doc_id][passage_id] = passage
                    else:
                        continue
            query_dfs.append(pd.DataFrame.from_records(queries))
            if di % 100 == 0:
                print(f"{round(100*di/len(docs), 2)}% done")
        json.dump(passages, open(passages_extract_path, "w"))
        pd.concat(query_dfs).to_csv(query_extract_path, index=False)


if __name__ == "__main__":
    import os

    CUR_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(CUR_DIR, "../../data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    EXTRACT_DATA_DIR = os.path.join(DATA_DIR, "extracted")
    passage_data_path = os.path.join(
        RAW_DATA_DIR, "WebAP/gradedText/grade.trectext_patched"
    )
    query_data_path = os.path.join(RAW_DATA_DIR, "WebAP/gradedText/gov2.query.json")

    query_filename = "webap_queries.csv"
    passage_filename = "webap_passages.json"
    extracted_query_path = os.path.join(EXTRACT_DATA_DIR, query_filename)
    extracted_passage_path = os.path.join(EXTRACT_DATA_DIR, passage_filename)
    # json.dump(de, open(extracted_passage_path, "w"))
    de = WebAPDataExtraction(query_data_path, passage_data_path)
    de.extract_data(extracted_query_path, extracted_passage_path, preprocess=True)
