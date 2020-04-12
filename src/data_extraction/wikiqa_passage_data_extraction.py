import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import json


class WikiQAPassageDataExtraction:
    def __init__(self, passage_data_path: str):
        self.passage_data_path = passage_data_path

    def load_passage_data(self):
        with open(self.passage_data_path) as f:
            data = json.load(f)
        return data

    def extract_passage_data(
        self, extract_path: str, stemmer: bool = None, lemmatizer: bool = None
    ):
        if (stemmer is not None) and (lemmatizer is not None):
            raise ValueError(
                """
                One of `stemmer` and `lemmatizer` must be None.
                Both cannot be applied
                """
            )

        punct_dict = str.maketrans("", "", string.punctuation)
        stop_word_list = stopwords.words("english")
        wordnet_lemmatizer = WordNetLemmatizer()
        porter = PorterStemmer()

        pass_dict = self.load_passage_data()
        n = len(list(pass_dict.keys()))
        for i, doc_id in enumerate(pass_dict.keys()):
            if i % 50 == 0:
                print(f"{(i/n)*100}% complete")
            for pass_id in pass_dict[doc_id].keys():
                passage = pass_dict[doc_id][pass_id]
                passage = passage.lower()
                passage = passage.translate(punct_dict)
                passage = nltk.word_tokenize(passage)
                passage = [word for word in passage if word not in stop_word_list]
                if lemmatizer is True:
                    passage = [wordnet_lemmatizer.lemmatize(word) for word in passage]
                elif stemmer is True:
                    passage = [porter.stem(word) for word in passage]
                pass_dict[doc_id][pass_id] = passage
        with open(extract_path, "w") as f:
            json.dump(pass_dict, f)


if __name__ == "__main__":
    import os

    CUR_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(CUR_DIR, "../../data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    EXTRACT_DATA_DIR = os.path.join(DATA_DIR, "extracted")
    # files to be extracted
    passage_data_path = os.path.join(
        RAW_DATA_DIR, "WikiPassageQA/document_passages.json"
    )

    de = WikiQAPassageDataExtraction(passage_data_path)

    pass_file = os.path.basename(passage_data_path)
    passage_extract_path = os.path.join(EXTRACT_DATA_DIR, pass_file)
    de.extract_passage_data(passage_extract_path, lemmatizer=True)
