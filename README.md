# Automatic Review Analyzer

The goal of this project is to design a classifier to use for sentiment analysis of product reviews. Our training set consists of reviews written by Amazon customers for various food products. The reviews, originally given on a 5 point scale, have been adjusted to a +1 or -1 scale, representing a positive or negative review, respectively.


![Alt Text](sentimentanalysistabledatasetexample.PNG)

In order to automatically analyze reviews, we will need to complete the following tasks:

-Implement and compare three types of linear classifiers: the perceptron algorithm, the average perceptron algorithm, and the Pegasos algorithm.

-Use your classifiers on the food review dataset, using some simple text features.

-Experiment with additional features and explore their impact on classifier performance.

# Data Description

## Overview
The data consists of several reviews, each labeled with -1 or +1 corresponding to a negative or positive review, respectively. It's split into four files:

- `reviews_train.tsv` (4000 examples)
- `reviews_validation.tsv` (500 examples)
- `reviews_test.tsv` (500 examples)

## Exploring the Data
To get a feel for the data, it's recommended to open the files using a text editor, spreadsheet program, or other scientific software like pandas.

## Translating Reviews to Feature Vectors
Reviews are translated into feature vectors using a bag of words approach. This involves compiling all words that appear in the training set into a dictionary, creating a list of unique words. Each review is then transformed into a feature vector of length d, with the ith coordinate set to 1 if the ith word in the dictionary appears in the review, or 0 otherwise. 

For example, given two documents "Mary loves apples" and "Red apples", with the dictionary {Mary; loves; apples; red}, the documents are represented as (1; 1; 1; 0) and (0; 0; 1; 1), respectively.

## Unigram vs. Bigram Models
A unigram model considers single words, while a bigram model includes phrases of length 2. For instance, in the bigram case, the dictionary might include "Mary loves" and "loves apples". For this project, only unigram word features are used.

## Implementation Details
- `utils.py` provides the `load_data` function to read .tsv files and return labels and texts.
- `project1.py` includes the `bag_of_words` function to generate a dictionary of unigram words.
- `extract_bow_feature_vectors` computes a feature matrix of ones and zeros from the raw data, using the resulting dictionary.
- Classification algorithms can then be applied using the feature matrix.




















Setup Details:

For this project we will be using Python 3.6 with some additional libraries. We strongly recommend that you take note of how the NumPy numerical library is used in the code provided. NumPy arrays are much more efficient than Python's native arrays when doing numerical computation. In addition, using NumPy will substantially reduce the lines of code you will need to write.

Note on software: For this project, we will need the NumPy numerical toolbox, and the matplotlib plotting toolbox.
