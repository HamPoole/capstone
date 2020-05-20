import itertools
import math

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
from gensim.models import Word2Vec
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

#Note: DVS stands for Document vectorization strategy and CA means Clustering algorithm.

# JC
#
# FM
#
# F1
#
# Beta - CV # measure
#
# silhouette # coefficient
#
# Kullback–Leibler # divergence


def preprocessing(text):
    text = re.sub(r"[{}]+".format(punctuation), '', text)
    tokens = word_tokenize(text.lower())
    StopWords = set(stopwords.words('english'))

    filtered_tokens = [token for token in tokens if token not in StopWords and re.match(r'^(-?\d+)(\.\d+)?$', token) == None and len(token) > 2]
    return ' '.join(filtered_tokens)


def top_tfidf_feats(row, terms, top_n=5):
    top_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [terms[i] for i in top_ids]
    return top_feats


def extract_tfidf_keywords(texts, top_n):
    tokenzier = word_tokenize
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=2000000,
                                       min_df=0, stop_words="english",
                                       use_idf=True, tokenizer=tokenzier, ngram_range=(1, 1))
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    terms = tfidf_vectorizer.get_feature_names()
    arr = []
    for i in range(0, tfidf_matrix.shape[0]):
        row = np.squeeze(tfidf_matrix[i].toarray())
        feats = top_tfidf_feats(row, terms, top_n)
        arr.append(feats)
    return arr


def compute_inertia(a, x):
    W = [np.mean(pairwise_distances(x[a == c, :])) for c in np.unique(a)]
    return np.mean(W)


def compute_gap(clustering, data, k_max=20, n_references=5):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    x = np.random.rand(*data.shape)
    reference_inertia = []
    for k in range(1, k_max+1):
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(x)
            local_inertia.append(compute_inertia(assignments, x))
        reference_inertia.append(np.mean(local_inertia))

    ondata_inertia = []
    for k in range(1, k_max+1):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))

    gap = np.log(reference_inertia)-np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)


def jaccard(labels1, labels2):
    """
    Computes the Jaccard similarity between two sets of clustering labels.
    The value returned is between 0 and 1, inclusively. A value of 1 indicates
    perfect agreement between two clustering algorithms, whereas a value of 0
    indicates no agreement. For details on the Jaccard index, see:
    http://en.wikipedia.org/wiki/Jaccard_index
    Example:
    labels1 = [1, 2, 2, 3]
    labels2 = [3, 4, 4, 4]
    print jaccard(labels1, labels2)
    @param labels1 iterable of cluster labels
    @param labels2 iterable of cluster labels
    @return the Jaccard similarity value
    """
    n11 = n10 = n01 = 0
    n = len(labels1)
    # TODO: Throw exception if len(labels1) != len(labels2)
    for i, j in itertools.combinations(range(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    print("11",n11)
    print("10",n10)
    print("01",n01)

    return float(n11) / (n11 + n10 + n01)


def FM(labels1, labels2):
    """
    Computes the Jaccard similarity between two sets of clustering labels.
    The value returned is between 0 and 1, inclusively. A value of 1 indicates
    perfect agreement between two clustering algorithms, whereas a value of 0
    indicates no agreement. For details on the Jaccard index, see:
    http://en.wikipedia.org/wiki/Jaccard_index
    Example:
    labels1 = [1, 2, 2, 3]
    labels2 = [3, 4, 4, 4]
    print jaccard(labels1, labels2)
    @param labels1 iterable of cluster labels
    @param labels2 iterable of cluster labels
    @return the Jaccard similarity value
    """
    n11 = n10 = n01 = 0
    n = len(labels1)
    # TODO: Throw exception if len(labels1) != len(labels2)
    for i, j in itertools.combinations(range(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    print("11",n11)
    print("10",n10)
    print("01",n01)

    return float(n11) / math.sqrt(n11*n11 + n11*n10 + n01*n10 + n11*n01)


def F1(labels1, labels2):
    n11 = n10 = n01 = 0
    n = len(labels1)
    # TODO: Throw exception if len(labels1) != len(labels2)
    for i, j in itertools.combinations(range(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    print("11",n11)
    print("10",n10)
    print("01",n01)

    return float(2*n11*n11) /(2*n11*n11 + n11*n10 + n11*n01)


def silhouette_coefficient1():
    print(1)
    #see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html