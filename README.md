# WikiPassageQA

Reproduction of "WikiPassageQA: A Benchmark Collection for Research on Non-factoid Answer Passage Retrieval" by Daniel Cohen, Liu Yang, and W. Croft (SIGIR18)

## Instructions for setting up

Download data from [here](https://ciir.cs.umass.edu/downloads/wikipassageqa/WikiPassageQA) and move it to `data/raw`.

Setup the dev environment by running:

1. `virtualenv -p python3 env`
2. `source env/bin/activate`
3. `pip install -r requirements.txt`

Installing nltk corpora:

1. `python -c"import nltk; nltk.download('stopwords')"`
2. `python -c"import nltk; nltk.download('wordnet')"`

## Contribution Guidelines

1. Clone the repository: `git clone git@github.com:nikhilsaldanha/WikiPassageQA.git`.
2. Setup the repository according to the instructions above.
3. Pick an open issue from the list [here](https://github.com/nikhilsaldanha/WikiPassageQA/issues) or create your own and assign it to yourself.
4. Create a branch to fix the issue: `git checkout -b <name-of-branch>`.
5. Once you're done, commit and push.
6. Go to the branch page and create a merge request. Ask a team member to approve changes.

## Data Extraction Pipeline

1. Passage Data Extraction
   - Convert to lower case
   - Remove punctuation
   - Tokenize
   - Remove stop words
   - Lemmatize/Stem

2. Query Data Extraction
   - Convert to lower case
   - Remove punctuation
   - Tokenize
   - Remove stop words
   - Lemmatize/Stem
   - Split each row with multiple comma separated passage ids in `RelevantPassages` column into multiple rows, each with 1 passage id.

**How to Run:**

1. Execute: `python src/data_extraction/query_data_extraction.py` to extract query data
2. Execute: `python src/data_extraction/passage_data_extraction.py` to extract passage data

Extracted data is stored in `data/extracted`. Query and Passage is converted to list of lemmatized/stemmed tokens.

## Feature Extraction Pipeline

1. Document Term Frequency
2. Collection Term Frequency

**How to Run:**
Execute `python src/feature_extraction/feature_extraction.py (train|test|dev)` to extract train, test or validation features

Extracted features are stored in `data/processed/train`, `data/processed/dev` and `data/processed/test`.

Structure of Collection Term Frequency `col_term_freq.json`:

```json
{
    "term1": 23,
    "term2": 31,
    ...
}
```

where each key is a unique term in the collection of documents and its value is the number of its occurances in the collection across all documents.

---

Structure of Document Term Frequency `doc_term_freq.json`:

```json
{
    "doc_id1": {
        "term1": 23,
        "term2": 31,
        ...
    },
    "doc_id2": {
        ...
    },
    ...
}
```

where each key is a unique id for a document in the collection and key is a dictionary of terms in the documents as keys and their frequency in that document as values.

## Model Creation Pipeline (TBD)

## Project Structure

```shell
data                   : all datasets(ignored from version control)
|__ processed          : features and pre-processed data
|   |__ features       : separate extracted features
|   |__ datasets       : combined features into datasets(train/valid/test)
|__ raw                : untouched data from the source

notebooks              : all notebooks (for quick and dirty work)
|__ data_analysis      : related to analysis and visualization of dataset
|__ feature_extraction : related to creating features
|__ models             : related to testing out models

src                    : all clean python scripts
|__ feature_extraction : one script for each feature
|__ models             : scripts for models
|__ experiments        : scripts to run all experiments (training, tuning, testing)

documents              : contains papers/reports required
```
