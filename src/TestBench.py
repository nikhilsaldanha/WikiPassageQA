
from experiments.TestMetrics import TestMetrics
import os
from models.tf_idf_vsm import tf_idf_VSM
import pandas as pd


def vsm(max_results):
    CUR_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(CUR_DIR, "../data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw/WikiPassageQA")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

    if os.path.exists(PROCESSED_DATA_DIR+"/vsm_test_results.csv"):
        test_result = pd.read_csv(PROCESSED_DATA_DIR+"/vsm_test_results.csv")
        return test_result


    testpath = os.path.join(CUR_DIR, '../data/raw/WikiPassageQA/test.tsv')
    test = pd.read_csv(testpath, sep='\t')

    model = tf_idf_VSM(PROCESSED_DATA_DIR+'/passage_df.csv')

    columns = ["QID", "DocID", "PassageID"]
    test_result = pd.DataFrame(columns=columns)

    length = test.shape[0]
    for i in range(length):
        result = model.get_ranked_passages(test.iloc[i, 1], max_results = max_results)
        #       DocId  PassageId                                            Passage  cosine_sim
        #12449    140          0  Proto-Slavic is defined as the last stage of t...    0.399138
        for j in range(result.shape[0]):
            df2 = pd.DataFrame([[ test.iloc[i,0], result.iloc[j,0], result.iloc[j,1] ]], columns=columns)
            test_result = test_result.append(df2)
        print("vsm: " + str(i) +"/"+str(length))

    test_result.to_csv(PROCESSED_DATA_DIR+"/vsm_test_results.csv", index=False)
    return test_result

if __name__ == "__main__":
    CUR_DIR = os.path.dirname(__file__)
    testpath = os.path.join(CUR_DIR, '../data/raw/WikiPassageQA/test.tsv')

    tester = TestMetrics(testpath)

    max_results = 20
    vsm_results = vsm(max_results)
    tester.get_metrics(vsm_results, max_results)
    