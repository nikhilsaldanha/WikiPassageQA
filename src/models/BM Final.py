import pandas as pd
import numpy as np
import string
import json
table = str.maketrans('', '', string.punctuation)
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from gensim.parsing.preprocessing import remove_stopwords
import re
import pickle
def word_count_pers(str,wordy):
    count=0
    str=remove_stopwords(str)
    wordy = wordnet_lemmatizer.lemmatize(wordy)
    words = word_tokenize(str)
    for word in words:
        word = wordnet_lemmatizer.lemmatize(word)
        if word==wordy:
            count+=1
    return count 
def pl(passage):
    passage=remove_stopwords(passage)
    words = word_tokenize(passage)
    return (len(words))
with open('../data/raw/WikiPassageQA/document_passages.json') as f:
    data = json.load(f)
df = pd.read_csv("../data/raw/extracted_query_data/train_exp.csv")
docids=[]
relid=[]
allpassages=[]
for i in data.keys():
    for j in data[i].keys():
        allpassages.append(data[i][j])
        docids.append(i)
        relid.append(j)
passages_df = pd.DataFrame(columns = ["DocId", "PassageId", "Passage"])        
passages_df["DocId"]= docids 
passages_df["PassageId"] =relid
passages_df["Passage"] = allpassages
#needs threading shawy
allpassages=[]
for i in data.keys():
    for j in data[i].keys():
        allpassages.append(data[i][j])
N=len(allpassages)
refinedpassages=[]
for i in allpassages:
    passage=re.sub(r'-|\n', ' ', i.translate(table).lower())
    refinedpassages.append(passage)
passage_lengths=[]
for i in refinedpassages:
    passage=remove_stopwords(i)
    words = word_tokenize(passage)
    passage_lengths.append(len(words))
avdl = np.array(passage_lengths).mean()
#stdl = np.array(passage_lengths).std()
uniqueQids= df['QID'].unique().tolist()
relpassages=dict()
for i in uniqueQids:
    relpassages[i]=[]    
for i in uniqueQids:
    for j in df.loc[df["QID"]==i].values:
        temp=[j[0],j[4]]
        relpassages[i].append(temp)    
def BM25(qid):
    k1=1.7
    k2=300
    b=0.75
    kk=k1*(1-b)
    question=df.loc[df["QID"]==qid]["Question"].values[0]
    question=re.sub(r'-', ' ', question)
    question=question.translate(table).lower()
    question=remove_stopwords(question)
    question_words= word_tokenize(question)
    input(question_words)
    R = len(relpassages[qid])
    ri=[]
    ni=[] 
    qi=[]
    for i in question_words:
        r=0
        for k in relpassages[qid]:
            tdocid=k[0]
            trelp =k[1]
            tpass = re.sub(r'\n|-', ' ', data[str(tdocid)][str(trelp)].translate(table).lower())
            if i in tpass:
                r+=1
        ri.append(r)
        count=0
        for passages in refinedpassages:
            if i in passages:
                count+=1
                continue
        ni.append(count)
        qi.append(word_count_pers(question,i))
    relevantindex=[]
    for j in range(len(refinedpassages)):
        for i in question_words:
            if i in refinedpassages[j]:
                relevantindex.append(j)
                break
    purepassages=[]
    for i in relevantindex:
        purepassages.append(refinedpassages[i])
    input(len(purepassages))
    BMsp=[]
    for i in purepassages:
        dl=pl(i)
        K = (kk + b*(dl/avdl))
        di=[]
        for w in question_words:
            di.append(word_count_pers(i,w))
        BMscore=0
        for i,j,k,l in zip(ri,ni,qi,di):
            a=(i+0.5)/(R-i+0.5)
            b=(j-i+0.5)/(N-j-R+i+0.5)
            c=(k1+1)*l/K+l
            d=(k2+1)*k/k2+k
            e=a/b
            score= np.log(e)*c*d
            BMscore+=score    
        BMsp.append(BMscore) 
    BMsp=(np.array(BMsp))
    sortedindices=BMsp.argsort()[-30:][::-1]
    final_output=[]
    for i in sortedindices:
        final_output.append([passages_df.loc[relevantindex[i]][0],passages_df.loc[relevantindex[i]][1]])
    return final_output    
BM25(3086),relpassages[3086] #still need to refine the number of pure passages.
