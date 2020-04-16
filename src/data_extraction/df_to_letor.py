import pandas as pd
import re
from sklearn.model_selection import train_test_split
import numpy as np


def df_to_letor(df, queries_df: pd.DataFrame) -> pd.DataFrame:
    # ensure that df has qid, docid, pid
    expected_cols = ("QID", "DocID", "PassageID")
    feat_cols = [col for col in df.columns if col not in expected_cols]
    assert set(expected_cols).issubset(
        set(df.columns)
    ), """DataFrame does not have some columns from ("QID", "DocID", "PassageID")"""

    def fn(row):
        a = queries_df[
            (queries_df["QID"] == row["QID"])
            & (queries_df["DocumentID"] == row["DocID"])
            & (queries_df["RelevantPassages"] == row["PassageID"])
        ]
        if a.shape[0] == 0:
            return 0
        else:
            return 1

    df["REL"] = df.apply(fn, axis=1)

    expected_cols = expected_cols + ("REL",)

    # initialize df
    letor_df = pd.DataFrame(columns=df.columns)
    i = 1
    for column in df.columns:
        print(column)
        if column == "QID":
            df[column] = [f"qid:{qid}" for qid in df[column]]
        elif column == "DocID":
            df[column] = [f"#docid = {docid}" for docid in df[column]]
        elif column == "PassageID":
            df[column] = [f"pid = {pid}" for pid in df[column]]
        elif column == "REL":
            pass
        else:
            df[column] = [f"{i}:{feat}" for feat in df[column]]
            i += 1
        letor_df[column] = df[column]
    letor_df = letor_df[["REL", "QID"] + feat_cols + ["DocID", "PassageID"]]
    return letor_df


if __name__ == "__main__":
    # df = pd.DataFrame(
    #     {
    #         "QID": [1, 2, 3, 884],
    #         "DocID": [11, 21, 23, 61],
    #         "PassageID": [3, 4, 5, 22],
    #         "f1": [33, 0, 2, 2],
    #         "f2": [33, 0, 2, 2],
    #         "f3": [33, 0, 2, 2],
    #     }
    # )

    # df1 = pd.read_csv("data/processed/L2R_features.csv", index_col=0)
    # df1 = df1.rename({"QId": "QID", "DocId": "DocID", "PassId": "PassageID"}, axis=1)
    # df2 = pd.read_csv("data/processed/ql_scores.csv", index_col=0)
    # df3 = pd.read_csv("data/processed/vsm_test_results_1000_dev.csv")

    df1 = pd.read_csv("data/processed/L2R_features_WebAP.csv", index_col=0)
    df1 = df1.rename({"QId": "QID", "DocId": "DocID", "PassId": "PassageID"}, axis=1)
    df2 = pd.read_csv("data/processed/ql_scores_webap.csv", index_col=0)
    df3 = pd.read_csv("data/processed/vsm_test_results_2000_WebAP.csv")

    df = pd.merge(df1, df2, how="inner", on=["DocID", "PassageID", "QID"])
    df = pd.merge(df, df3, how="inner", on=["DocID", "PassageID", "QID"])

    queries_df = pd.read_csv("data/extracted/webap_queries.csv")
    letor_df = df_to_letor(df, queries_df)
    letor_df.to_csv("data/processed/letor.csv", index=False)
    letor_df = pd.read_csv("data/processed/letor.csv")
    train_non_rel, test_non_rel = train_test_split(
        letor_df[letor_df["REL"] == 0].to_numpy(), test_size=0.2, random_state=42
    )

    train_rel, test_rel = train_test_split(
        letor_df[letor_df["REL"] == 1].to_numpy(), test_size=0.2, random_state=42
    )

    ncols = letor_df.shape[1]

    train_df = pd.DataFrame(
        np.concatenate((train_non_rel, train_rel), axis=0),
        columns=[f"{i}" for i in range(ncols)],
    )
    test_df = pd.DataFrame(
        np.concatenate((test_non_rel, test_rel), axis=0),
        columns=[f"{i}" for i in range(ncols)],
    )

    train_df = train_df.sort_values(by="1", axis=0)
    test_df = test_df.sort_values(by="1", axis=0)
    train_txt = re.sub(r" +", " ", train_df.to_string(header=False, index=False))
    test_txt = re.sub(r" +", " ", test_df.to_string(header=False, index=False))
    with open("data/processed/l2r_webap_test.txt", "w") as f:
        f.write(test_txt)

    with open("data/processed/l2r_webap_train.txt", "w") as f:
        f.write(train_txt)
