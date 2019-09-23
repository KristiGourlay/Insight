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

x_train = x_train.astype('U')
x_test = x_test.astype('U')

tvec = TfidfVectorizer(stop_words='english',
                        ngram_range=(1, 4),
                        encoding='utf-8')







train_data_tfid = tvec.fit_transform(x_train.apply(lambda x: np.str_(x)))

test_data_tfid = tvec.transform(x_test.apply(lambda x: np.str_(x)))

model = LogisticRegression(class_weight='balanced')
model.fit(train_data_tfid, y_train)
print(model.score(train_data_tfid, y_train))
print(model.score(test_data_tfid, y_test))


predictions = model.predict(test_data_tfid)

pd.DataFrame(predictions, y_test)


# Creating pipeline and pickleing

tvec_pipe2 = Pipeline([
    ('tfidf', tvec),
    ('model', model)
])



tvec_pipe2.fit(x_train, y_train)


pickle.dump(tvec_pipe2, open('../projectname/bots_and_pickles/tvec_model2.pkl', 'wb'))
