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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE


df = pd.read_csv('../data/cleaned/book_df.csv', index_col=0)
df = df.drop_duplicates(subset=['text'])
# TRAIN TEST SPLIT
x = df['text']
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.9, random_state=42, shuffle=True, stratify=y)

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

#GridSearchCV


lr = LogisticRegression()

lr_params = {
    'solver': ['lbfgs', 'sag'],
    'class_weight': ['balanced'],
    'penalty': ['l2'],
    'C': [.1, .01, .001, .001, .2, .02, .002],
    'multi_class': ['multinomial', 'auto']
}

gs = GridSearchCV(lr, param_grid=lr_params)
gs.fit(train_data_features, y_train)

gs.best_params_






# Logistic Regression with CountVectorizer
model = LogisticRegression()
model.fit(train_data_features, y_train)

model.score(train_data_features, y_train)

def classification_metrics(y_test, y_pred):
    print(f' Accuracy Score: {accuracy_score(y_test, preds)}')
    print(f' Precision Score: {precision_score(y_test, preds, average = None)}')
    print(f' Recall Score: {recall_score(y_test, preds, average = None)}')

classification_metrics(test_data_features, y_test)



# Imbalance learn and regularization
sm = SMOTE()
x_reb, y_reb = sm.fit_sample(train_data_features, y_train)


log_reg = LogisticRegression(C=.01, class_weight='balanced')
log_reg.fit(x_reb, y_reb)

cross_val_score(log_reg, x_reb, y_reb, cv=5).mean()

print(log_reg.score(x_reb, y_reb))
print(log_reg.score(test_data_features, y_test))

preds = log_reg.predict(test_data_features)




classification_metrics(y_test, preds)




#Comparing prediction to test targets

comparison = pd.DataFrame(preds, y_test)


comparison = comparison.reset_index()
comparison = comparison.rename(columns={0: 'prediction'})
comparison['prediction'] = comparison['prediction'].astype(int)


comparison['correct'] = np.where(comparison['target'] == comparison['prediction'], 1, 0)
comparison = comparison.rename(columns={0: 'prediction'})
comparison.target = comparison.target.astype(int)
comparison.tail(10)

comparison['difference'] = abs(comparison.target - comparison.prediction)
comparison.head(10)

borderline_dates = len(comparison[comparison['difference'] <= 1]) / len(comparison['difference'])
borderline_dates



#pipeline

cvec_pipe = Pipeline([
    ('cvec', cvec),
    ('model', log_reg)
])



cvec_pipe.fit(x_train, y_train)


pickle.dump(cvec_pipe, open('../projectname/bots_and_pickles/cvec_model.pkl', 'wb'))




# Random Forest Classifier with CountVectorizer

rf = RandomForestClassifier(n_estimators=50)

cross_val_score(rf, x_reb, y_reb, cv=5).mean()


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
