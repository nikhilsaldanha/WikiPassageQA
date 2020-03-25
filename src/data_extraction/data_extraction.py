import pandas as pd
import numpy as np


class DataExtraction:
    def __init__(self, query_data_path: str):
        self.query_data_path = query_data_path

    def load_query_data(self):
        # load tsv file
        df = pd.read_csv(self.query_data_path, delimiter="\t")
        return df

    def split_rows(self, df: pd.DataFrame, col: str):
        # column with comma separated values
        lst_col = col
        # all other columns except for `lst_col`
        oth_cols = df.columns.difference([lst_col])

        # convert df with comma separated values
        # to list of values
        df_lst_col = df.assign(**{lst_col: df[lst_col].str.split(",")})

        # dict with key as column name and value as list of row values
        # to store the row expanded df
        expanded_dict = {}

        # iterate through `oth_cols` and created expanded dict
        for col in oth_cols:
            expanded_dict.update(
                {
                    # repeat each row list length number of times
                    col: np.repeat(
                        df_lst_col[col].values, df_lst_col[lst_col].str.len()
                    )
                }
            )

        # concatenate together all `lst_col` values
        lst_col_dict = {lst_col: np.concatenate(df_lst_col[lst_col].values)}

        # create 1 dataframe with split rows
        split_df = pd.DataFrame(expanded_dict).assign(**lst_col_dict)

        return split_df

    def extract_query_data(self, extract_path):
        query_data = self.load_query_data()
        split_df = self.split_rows(query_data, "RelevantPassages")
        split_df.to_csv(extract_path, index=False)


if __name__ == "__main__":
    import os

    CUR_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(CUR_DIR, "../../data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

    # files to be extracted
    query_data_paths = [
        os.path.join(RAW_DATA_DIR, "WikiPassageQA/test.tsv"),
        os.path.join(RAW_DATA_DIR, "WikiPassageQA/train.tsv"),
        os.path.join(RAW_DATA_DIR, "WikiPassageQA/dev.tsv"),
    ]

    for query_data_path in query_data_paths:
        extract_data_path = os.path.join(RAW_DATA_DIR, "extracted_query_data")

        de = DataExtraction(query_data_path)

        query_file = os.path.basename(query_data_path).split(".")[0]
        query_extract_path = os.path.join(extract_data_path, f"{query_file}_exp.csv")
        de.extract_query_data(query_extract_path)
        print(pd.read_csv(open(query_extract_path, "r")))

    # test for split_rows
    # df = pd.DataFrame(
    #     [
    #         {"var1": "a,b,c", "var2": 1, "var3": "XX"},
    #         {"var1": "d,e,f,x,y", "var2": 2, "var3": "ZZ"},
    #     ]
    # )
    # split_df = DataExtraction("").split_rows(df, "var1")
    # print(split_df)
