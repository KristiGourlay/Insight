import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import pickle

from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline



df = pd.read_csv('../data/cleaned/book_df_cleaned.csv', index_col=0)

# TRAIN TEST SPLIT
x = df['text']
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, random_state=42, shuffle=True, stratify=y)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# CountVectorizer

cvec = CountVectorizer(stop_words='english',
                        lowercase=True,
                        ngram_range=(1, 5),
                        strip_accents='unicode')

train_data_features = cvec.fit_transform(x_train.apply(lambda x: np.str_(x)))

train_data_features

train_data_features.todense()

df_cv = pd.DataFrame(train_data_features.todense(), columns=cvec.get_feature_names())

df_cv.head()

test_data_features = cvec.transform(x_test.apply(lambda x: np.str_(x)))

test_data_features.todense()



# Logistic Regression with CountVectorizer

log_reg = LogisticRegression()
log_reg.fit(train_data_features, y_train)
print(log_reg.score(train_data_features, y_train))
print(log_reg.score(test_data_features, y_test))



# Random Forest Classifier with CountVectorizer

rf = RandomForestClassifier(n_estimators=100)

cross_val_score(rf, train_data_features, y_train, cv=5).mean()


# GridSearch over RandomForestClassifier

features = train_data_features.shape
np.sqrt(features)

rf = RandomForestClassifier()
rf_params = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'max_depth': [None, 1, 2, 3, 4, 5],
    'min_samples_split': [2, 3, 4]
}

gs = GridSearchCV(rf, param_grid=rf_params)
gs.fit(train_data_features, y_train)
print(gs.best_score_)
gs.best_params_



# TF-IDF Vectorizer

x_train = x_train.astype('U')
x_test = x_test.astype('U') #documentation says that the dtype must be converted to unicode



tvec = TfidfVectorizer(stop_words='english',
                        ngram_range=(1, 4))


train_data_tfid = tvec.fit_transform(x_train.apply(lambda x: np.str_(x)))

test_data_tfid = tvec.transform(x_test.apply(lambda x: np.str_(x)))

model = LogisticRegression()
model.fit(train_data_tfid, y_train)
print(model.score(train_data_tfid, y_train))
print(model.score(test_data_tfid, y_test))


predictions = model.predict(test_data_tfid)

pd.DataFrame(predictions, y_test)


# Creating pipeline and pickleing

steps = [
    ('tfid_vectorize', tvec),
    ('model', model)
]

pipe = Pipeline(steps=steps)

pipe.fit(x_train, y_train)

preds = pipe.predict(x_test)

pickle.dump(pipe, open('../projectname/tvec_model.pkl', 'wb'))


# Catboost with TFIDF

import catboost as cb
from catboost import CatBoostClassifier
cb = CatBoostClassifier(iterations=100, learning_rate=0.5, early_stopping_rounds=20, loss_function='MultiClass',
                    custom_metric="Accuracy")

cb.fit(train_data_tfid, y_train, eval_set=(test_data_tfid, y_test), plot=True)

ftrs = cb.get_feature_importance(prettified=True)

{feature_name : value for feature_name, value in ftrs}

tuned_model = CatBoostClassifier(
    random_seed=63,
    iterations=1000,
    learning_rate=0.03,
    l2_leaf_reg=3,
    bagging_temperature=1,
    random_strength=1,
    one_hot_max_size=2,
    leaf_estimation_method='Newton'
)

tuned_model.fit(
    train_data_tfid, y_train,
    verbose=False,
    eval_set=(test_data_tfid, y_test),
    plot=True
)

best_model = CatBoostClassifier(
    random_seed=63,
    iterations=int(tuned_model.tree_count_ * 1.2),
)
best_model.fit(
    train_data_tfid, y_train,
    verbose=100
)


predictions = model.predict(test_data_tfid)

catboost_test = pd.DataFrame(predictions, y_test)
catboost_test.to_csv('cb_test_results.csv')


catboost_results = pd.DataFrame(predictions, y_train)
catboost_results.to_csv('cb_results.csv')


pickle.dump(best_model, open('cb_model.pkl', 'wb'))

md = pickle.load(open('cb_model.pkl', 'rb'))


md.predict(test_data_tfid)
md.score(test_data_tfid, y_test)
md.score(train_data_tfid, y_train)


#Pipeline for CatBoostClassifier with TF-IDF

cb_model = pickle.load(open('cb_model.pkl', 'rb'))


steps = [
    ('tfid_vectorize', tvec),
    ('model', cb_model)
]

pipe2 = Pipeline(steps=steps)

pipe2.fit(x_train, y_train)

preds = pipe2.predict(x_test)
