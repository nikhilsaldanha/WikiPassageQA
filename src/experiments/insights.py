import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes

if __name__ == "__main__":

    # WikiPassageQA
    data = pd.read_csv("./data/raw/WikiPassageQA/test.tsv", sep='\t')
    data = data.rename(columns={"RelevantPassages":"PassageID"})
    split_passageIds = data["PassageID"].astype(str).str.split(",")
    columns = ["QID", "DocID", "PassageID"]
    test_true = pd.DataFrame(columns=columns)

    for i in range(split_passageIds.size):
        for j in range(len(split_passageIds.iloc[i])):
            df2 = pd.DataFrame([[ data.iloc[i,0], data.iloc[i,2], split_passageIds.iloc[i][j] ]], columns=columns)
            test_true = test_true.append(df2)

    # convert char Passage Ids to int
    test_true["PassageID"] = test_true["PassageID"].astype(int)
    freq = test_true['QID'].value_counts().values
    plt.figure(figsize=(10,6))
    plt.hist(freq)
    plt.title("WikiPassageQA query relevent documents count Histogram")
    plt.savefig("./images/wikiQA_rel_freq.png")

    #------------------------------------------------------------------
    data = pd.read_csv("./data/processed/passage_df_WikiQA.csv")
    freq = data["Passage"].str.len()
    plt.figure(figsize=(10,6))
    plt.hist(freq, bins=100, range=(0,2500))


    plt.title("WikiPassageQA Passage Lengths Histogram")
    plt.legend()
    plt.savefig("./images/wikiQA_passage_length.png")





    #==========================================================================

    # WebAP
    data = pd.read_csv("./data/extracted/webap_queries.csv")
    freq = data['QID'].value_counts().values
    print(freq)
    plt.figure(figsize=(10,6))
    plt.hist(freq)
    plt.title("WebAP query relevent documents count Histogram")
    plt.savefig("./images/wepap_rel_freq.png")

    data = pd.read_csv("./data/extracted/webap_queries.csv")
    df = pd.DataFrame(data.groupby(['QID', 'Question']).size().rename("ReleventPassagesCount"))
    print(df)

    #------------------------------------------------------------------

    data = pd.read_csv("./data/processed/passage_df_WebAP.csv")
    freq = data["Passage"].astype(str).str.len()
    plt.figure(figsize=(16,9))
    plt.hist(freq, bins=100, histtype="step")

    plt.title("WikiPassageQA Passage Lengths Histogram")
    plt.legend()
    plt.savefig("./images/webAP_passage_length.png")