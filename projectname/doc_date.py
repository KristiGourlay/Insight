import pandas as pd
import flask
from flask import render_template, request
import json
import pickle
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# from bots_and_pickles.language_transformer import LanguageTransformer

pipe = pickle.load(open('bots_and_pickles/tvec_model2.pkl', 'rb'))


def clean_text(raw_text):
    lemmatizer = WordNetLemmatizer()
    raw_text = str(raw_text)
    lower_case = raw_text.lower()
    retokenizer = RegexpTokenizer(r'[a-z]+')
    words = retokenizer.tokenize(lower_case)

    return(lemmatizer.lemmatize(" ".join(words)))



def decoder(prediction):
    if prediction == 0:
        return('The Renaissance (Before 1670)')
    if prediction == 1:
        return('The Restoration and the Age of Enlightenment (1670-1830)')
    if prediction == 2:
        return('The Romantic Period (1830-1870)')
    if prediction == 3:
        return('The Late Victorian Era (Naturalism and Realism) (1870-1910)')
    if prediction == 4:
        return('The Modernist Period (1910-1945)')
    if prediction == 5:
        return('The Contemporary Period (1945-present)')
    else:
        return('Try Again')



def template_chooser(prediction):
    if prediction == 0:
        template = 'result0.html'
    elif prediction == 1:
        template = 'result1.html'
    elif prediction == 2:
        template = 'result2.html'
    elif prediction == 3:
        template = 'result3.html'
    elif prediction == 4:
        template = 'result4.html'
    elif prediction == 5:
        template = 'result5.html'
    else:
        template = 'home.html'

    return template



app = flask.Flask(__name__)

@app.route('/home')
def home():
   with open("templates/home.html", 'r') as home:
       return home.read()

@app.route('/result', methods=['POST', 'GET'])
def result():
    if flask.request.method == 'POST':

        inputs = flask.request.form

        results = inputs['text']
        text = [clean_text(results)] #needs to be in a list
        prediction = pipe.predict(text)
        prediction = list(prediction)[0] #if not, prediction will be an array of the prediction number instead of a single number target
        result = decoder(prediction)

        template = template_chooser(prediction)

        return render_template(template, result=result)



if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '0.0.0.0'
    # HOST = '127.0.0.1'
    PORT = 5000
    debug = True

    app.run(HOST, PORT)
