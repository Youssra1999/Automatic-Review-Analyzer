# Automatic Review Analyzer

## Introduction
The goal of this project is to design a classifier for sentiment analysis of product reviews. Our objective is to classify reviews written by Amazon customers for various food products as either positive or negative.

## Data Description

### Overview
The dataset consists of several reviews, each labeled with -1 or +1 corresponding to a negative or positive review, respectively. It's split into four files:

- `reviews_train.tsv` (4000 examples)
- `reviews_validation.tsv` (500 examples)
- `reviews_test.tsv` (500 examples)

### Exploring the Data
To understand the dataset better, it's recommended to open the files using a text editor, spreadsheet program, or other scientific software like pandas.

### Translating Reviews to Feature Vectors
We'll convert reviews into feature vectors using a bag of words approach. This involves compiling all words appearing in the training set into a dictionary. Each review is then transformed into a feature vector of length d. For example, given two documents "Mary loves apples" and "Red apples", with the dictionary {Mary; loves; apples; red}, the documents are represented as (1; 1; 1; 0) and (0; 0; 1; 1), respectively.

### Unigram vs. Bigram Models
We'll focus on unigram word features for this project. However, the bag of words model can be extended to include phrases of length m (bigram model).

## Implementation Details
- `utils.py` provides the `load_data` function to read .tsv files and return labels and texts.
- `project1.py` includes the `bag_of_words` function to generate a dictionary of unigram words.
- `extract_bow_feature_vectors` computes a feature matrix of ones and zeros from the raw data, using the resulting dictionary.

## Setup Details
For this project, we'll be using Python 3.6 with the NumPy and matplotlib libraries. Ensure you have these libraries installed to run the code effectively.

