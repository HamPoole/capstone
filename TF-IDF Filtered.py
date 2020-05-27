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

StopWords = set(stopwords.words('english'))
porter = PorterStemmer()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def preprocessing(text):
    text = re.sub(r"[{}]+".format(punctuation), '', text)
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in StopWords and re.match(r'^(-?\d+)(\.\d+)?$', token) == None and len(token) > 2]
    return ' '.join(filtered_tokens)

def top_tfidf_feats(row, terms, top_n=25):
    top_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [terms[i] for i in top_ids]
    return top_feats


def extract_tfidf_keywords(texts, top_n=25):
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

Raw = [line.strip('\n') for line in open('1.Raw Input.txt').readlines()]
Cleaned_Corpus = [preprocessing(line) for line in Raw]
TF_Words = extract_tfidf_keywords(Cleaned_Corpus, 20)

model = Word2Vec([word_tokenize(line) for line in Cleaned_Corpus], size=100, window=5, min_count=1, workers=4)
doc_vec = []
for f in TF_Words:
    doc_vec.append(np.mean(model[f], axis=0))
X = pd.DataFrame(doc_vec)
X.to_excel('4.TF-IDF Filtered Version.xlsx')