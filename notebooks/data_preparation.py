import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE

df = pd.read_csv('../data/processed/data2.csv', index_col=0)
manual_df = pd.read_csv('../data/processed/manual_df.csv')

df = df.append(manual_df)


# SETTING TARGETS

df['date'] = pd.to_numeric(df['date'], errors='coerce')

df = df.dropna(subset=['date'])
df['date'] = df['date'].astype(int)


##Time Periods
# The Renaissance (before 1670)
# The Restoration and The Age of the ENlightenment(1670-1800)
# The Romantic Period (1830-1870)
# Realism and Naturalism [Victorian] (1870-1920)
# Modernist Period [Modernism ](1920-1945)
# Contemporary Period [Post-Modernism](1945-present)

bins = [0, 1670, 1800, 1870, 1920, 1945, np.inf]
names = [0, 1, 2, 3, 4, 5]

df['target'] = pd.cut(df['date'], bins, labels=names)


df.groupby('target').count()

df.head()
#attempt to balance imnbalance target (temporary )

count_class_0, count_class_1, count_class_2, count_class_3, count_class_4, count_class_5 = df.target.value_counts()

df_class_0 = df[df['target'] == 3]
df_class_1 = df[df['target'] == 2]
df_class_2 = df[df['target'] == 5]
df_class_3 = df[df['target'] == 1]
df_class_4 = df[df['target'] == 4]
df_class_5 = df[df['target'] == 0]

df_class_5 = df_class_5.sample(count_class_2, replace=True)
df_class_0 = df_class_0.sample(count_class_2, replace=True)
df_class_1 = df_class_1.sample(count_class_2, replace=True)
df_class_3 = df_class_3.sample(count_class_2, replace=True)
df_class_4 = df_class_4.sample(count_class_2, replace=True)


df_text_over = pd.concat([df_class_0, df_class_1, df_class_2, df_class_3, df_class_4, df_class_5])

print(df.target.value_counts())
print(df_text_over.target.value_counts())

df = df_text_over

df.head()

df.to_csv('../data/cleaned/book_df.csv') # version before cleaning


# DATA CLEANING


 lemmatizer = WordNetLemmatizer()
 st = df['text'].tolist()


 def clean_text(raw_text):
     lower_case = raw_text.lower()
     retokenizer = RegexpTokenizer(r'[a-z]+')
     words = retokenizer.tokenize(lower_case)
     return(lemmatizer.lemmatize(" ".join(words)))

 num_excerpts = df['text'].size

 clean_text_excerpts = []

 for i in range(0, num_excerpts):
     clean_text_excerpts.append( clean_text( st[i] ))


 df['text'] = clean_text_excerpts



df.to_csv('../data/cleaned/book_df_cleaned.csv') # version after cleaning
