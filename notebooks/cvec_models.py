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


df = pd.read_csv('../data/cleaned/final_df.csv', index_col=0)

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
                        ngram_range=(1, 3),
                        strip_accents='unicode')

train_data_features = cvec.fit_transform(x_train.apply(lambda x: np.str_(x)))


test_data_features = cvec.transform(x_test.apply(lambda x: np.str_(x)))





# Logistic Regression with CountVectorizer

log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(train_data_features, y_train)
print(log_reg.score(train_data_features, y_train))
print(log_reg.score(test_data_features, y_test))

preds = log_reg.predict(test_data_features)


#GridSearchCV


lr = LogisticRegression()

lr_params = {
    'solver': ['lbfgs', 'sag'],
    'class_weight': ['balanced'],
    'multi_class': ['multinomial', 'auto']
}

gs = GridSearchCV(lr, param_grid=lr_params)
gs.fit(train_data_features, y_train)
print(gs.best_score_)
gs.best_params_




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
