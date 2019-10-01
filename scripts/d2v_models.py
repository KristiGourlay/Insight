import pandas as pd
import numpy as np
import re
import multiprocessing
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import catboost as cb
from catboost import CatBoostClassifier

from nltk.tokenize import RegexpTokenizer
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import utils

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('../data/cleaned/book_df.csv', index_col=0)
df.head(2)

df.target.value_counts()
df = df.drop_duplicates(subset=['info'])

# FURTHER CLEANING

lemmatizer = WordNetLemmatizer()
st = df['text'].tolist()


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
df['text'] = df['text'].apply(tokenize_text)

df['text'] = df['text'].astype('str')

# Doc2Vec

train, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df.target)

train_tagged = train.apply(
    lambda r: TaggedDocument(words=(r['text']), tags=[r.target]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=(r['text']), tags=[r.target]), axis=1)

cores = multiprocessing.cpu_count()


# Doc2Vec: Distributed Bag of Words model

wdv_model = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=1, sample = 0, workers=cores)

wdv_model.build_vocab([x for x in tqdm(train_tagged.values)])



for epoch in range(30):
    wdv_model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    wdv_model.alpha -= 0.002
    wdv_model.min_alpha = wdv_model.alpha



def vector_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors



y_train, X_train = vector_for_learning(wdv_model, train_tagged)
y_test, X_test = vector_for_learning(wdv_model, test_tagged)
logreg = LogisticRegression(C=.001, solver='lbfgs', class_weight='balanced')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


logreg.score(X_train, y_train)
logreg.score(X_test, y_test)






# Distributed Memory model (Doc2Vec)

dmm_model = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
dmm_model.build_vocab([x for x in tqdm(train_tagged.values)])


for epoch in range(30):
    dmm_model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    dmm_model.alpha -= 0.002
    dmm_model.min_alpha = dmm_model.alpha


y_train, X_train = vector_for_learning(dmm_model, train_tagged)
y_test, X_test = vector_for_learning(dmm_model, test_tagged)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

logreg.score(X_train, y_train)
logreg.score(X_test, y_test)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))


wdv_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
dmm_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)




# Concatenated Model (Doc2Vec)


new_model = ConcatenatedDoc2Vec([wdv_model, dmm_model])

def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_train, X_train = get_vectors(new_model, train_tagged)
y_test, X_test = get_vectors(new_model, test_tagged)

from imblearn.over_sampling import SMOTE

logreg = LogisticRegression(C=.01, class_weight='balanced')
sm = SMOTE()
X_train, y_train = sm.fit_sample(X_train, y_train)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
logreg.score(X_train, y_train)
logreg.score(X_test, y_test)


y_pred


# Creating polarity(-1 1) for vectors created

doc_2_vec = []


corpus = df['text']

for c in corpus:
    doc_2_vec.append(wdv_model.infer_vector(c)[0])

df['doc_2_vec'] = doc_2_vec


df.head()
# df.to_csv('../data/cleaned/df_with_sentiment_doc2vec.csv')
