import pandas as pd
import flask
from flask import render_template, request
import json
import pickle
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from bots_and_pickles.language_transformer import LanguageTransformer
from bots_and_pickles.template_renderer import TemplateRenderer

pipe = pickle.load(open('bots_and_pickles/tvec_model2.pkl', 'rb'))




app = flask.Flask(__name__)

@app.route('/home')
def home():
   with open("templates/home.html", 'r') as home:
       return home.read()

@app.route('/about')
def about():
    with open('templates/about.html', 'r') as about:
        return about.read()

@app.route('/result', methods=['POST', 'GET'])
def result():
    if flask.request.method == 'POST':

        inputs = flask.request.form
        results = inputs['text']
        
        ct = LanguageTransformer()
        excerpt = [ct.clean_text(results)] #needs to be in a list
        prediction = pipe.predict(excerpt)
        prediction = list(prediction)[0] #if not, prediction will be an array of the prediction number instead of a single number target

        tr = TemplateRenderer()
        result = tr.decoder(prediction)
        template = tr.template_chooser(prediction)


        return render_template(template, result=result)



if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '0.0.0.0'
    # HOST = '127.0.0.1'
    PORT = 5000
    debug = True

    app.run(HOST, PORT)
