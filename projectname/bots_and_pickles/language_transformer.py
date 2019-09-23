from sklearn.base import TransformerMixin
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class LanguageTransformer(TransformerMixin):

    def fit(self, x_train):
        return self

    def transform(self, x_train):
        new_list = []
        new_line = []
        final_line = []
        final_entry = []
        for item in x_train:
            new_list.append(item + ', ')
            for list_item in new_list:
                new_line.append(list_item.split())
                for line in new_line:
                    final_line = []
                    for word in line:
                        lemmatizer = WordNetLemmatizer()
                        raw_text = str(word)
                        string_lower_case = raw_text.lower()
                        # new_text = string_lower_case.astype('U')
                        retokenizer = RegexpTokenizer(r'[a-z]+')
                        words = retokenizer.tokenize(string_lower_case)
                        lemm_words = lemmatizer.lemmatize(" ".join(words))
                        final_line.append(lemm_words)

                final_entry.append(final_line)


        return final_entry
