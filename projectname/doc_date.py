import pandas as pd
import flask
from flask import render_template, request
import json
import pickle


pipe = pickle.load(open('tvec_model.pkl', 'rb'))


def decoder(prediction):
    if prediction == 0:
        return('The Renaissance (Before 1670)')
    if prediction == 1:
        return('The Restoration and the Age of Enlightenment (1670-1800)')
    if prediction == 2:
        return('The Romantic Period (1830-1870)')
    if prediction == 3:
        return('The Late Victorian Era (Naturalism and Realism) (1870-1920)')
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
        text = [results] #needs to be in a list
        prediction = pipe.predict(text)
        prediction = list(prediction)[0] #if not, prediction will be an array of the prediction number instead of a single number target
        result = decoder(prediction)

        template = template_chooser(prediction)

        return render_template(template, result=result)



if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = 4000

    app.run(HOST, PORT)
