import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import pickle

import catboost as cb
from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline


df = pd.read_csv('../data/cleaned/final_df.csv', index_col=0)

# TRAIN TEST SPLIT
x = df['text']
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, random_state=42, shuffle=True, stratify=y)

tvec = TfidfVectorizer(stop_words='english',
                        ngram_range=(1, 4),
                        encoding='utf-8')


train_data_tfid = tvec.fit_transform(x_train.apply(lambda x: np.str_(x)))

test_data_tfid = tvec.transform(x_test.apply(lambda x: np.str_(x)))



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
    leaf_estimation_method='Newton',
    early_stopping_rounds=20,
    loss_function='MultiClass'
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



preds = best_model.predict(test_data_tfid)
best_model.score(test_data_tfid, y_test)


# Creating pipeline and pickleing

tvec_cat_pipe1 = Pipeline([
    ('tfidf', tvec),
    ('model', best_model)
])



tvec_cat_pipel.fit(x_train, y_train)
