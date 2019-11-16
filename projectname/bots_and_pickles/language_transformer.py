from sklearn.base import TransformerMixin
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class LanguageTransformer(TransformerMixin):

    def fit(self, raw_text):
        return self

    def clean_text(self, raw_text):

        raw_text = str(raw_text)
        lower_case = raw_text.lower()
        retokenizer = RegexpTokenizer(r'[a-z]+')
        words = retokenizer.tokenize(lower_case)
        new_words = " ".join(words)
        return new_words

    def lemm(self, new_words):
        lemmatizer = WordNetLemmatizer()

        final_sentences = []
        for line in new_words:
            line = str(line)
            line = line.split()
            new_sentence = []
            for word in line:
                new_word = lemmatizer.lemmatize(word)
                new_sentence.append(new_word)
            final_sentences.append(new_sentence)

        return final_sentences
