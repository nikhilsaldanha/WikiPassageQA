# WikiPassageQA

Reproduction of "WikiPassageQA: A Benchmark Collection for Research on Non-factoid Answer Passage Retrieval" by Daniel Cohen, Liu Yang, and W. Croft (SIGIR18)

## Instructions for setting up

Download data from [here](https://ciir.cs.umass.edu/downloads/wikipassageqa/WikiPassageQA) and move it to `data/raw`.

Setup the dev environment by running:

1. `virtualenv -p python3 env`
2. `source env/bin/activate`
3. `pip install -r requirements.txt`

## Contribution Guidelines

1. Clone the repository: `git clone git@github.com:nikhilsaldanha/WikiPassageQA.git`.
2. Setup the repository according to the instructions above.
3. Pick an open issue from the list [here](https://github.com/nikhilsaldanha/WikiPassageQA/issues) or create your own and assign it to yourself.
4. Create a branch to fix the issue: `git checkout -b <name-of-branch>`.
5. Once you're done, commit and push.
6. Go to the branch page and create a merge request. Ask a team member to approve changes.

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
