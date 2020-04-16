import json
import pandas as pd
import numpy as np
import threading


def thread_function(i, till_len):
    global passages_dict
    global doc_ids

    column_names = ["DocId", "PassageId", "Passage"]
    passages_df = pd.DataFrame(columns=column_names)
    count = 0
    for j in range(i*50, i*50 + till_len):
        for pass_id in passages_dict[doc_ids[j]]:
            passage = passages_dict[doc_ids[j]][pass_id]
            df2 = pd.DataFrame([[doc_ids[j], pass_id , passage]], columns=column_names)
            passages_df = passages_df.append(df2)
        count = count + 1
        print ("Thread " + str(i) +" docs completed: " + str(count))
    
    passages_df.to_csv(PROCESSED_DATA_DIR+'/passage_df_'+str(i)+'_.csv', index=False)

if __name__ == "__main__":
    import os

    CUR_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(CUR_DIR, "../../data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw/WikiPassageQA")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

    passages_path = RAW_DATA_DIR+ "/document_passages.json"


    with open(passages_path, 'r') as f:
        passages_dict = json.load(f)
        column_names = ["DocId", "PassageId", "Passage"]
        passages_df = pd.DataFrame(columns=column_names)

        doc_ids = []
        for doc in passages_dict:
            dos_ids = doc_ids.append(doc)

        ts = []

        for i in range(17):
            till_len = 50
            if i == 16:
                till_len = 63
            x = threading.Thread(target=thread_function, args=(i, till_len))
            ts.append(x)

        for t in ts:
            t.start()
        
        for t in ts:
            t.join()

        # join the data frames
        for i in range(17):
            df = pd.read_csv(PROCESSED_DATA_DIR+'/passage_df_'+str(i)+'_.csv')
            print(df.shape)
            passages_df = passages_df.append(df)

        print(passages_df.shape)
        passages_df.to_csv(PROCESSED_DATA_DIR+'/passage_df.csv', index=False)


