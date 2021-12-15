from types import MethodDescriptorType
from flask import Flask, redirect, url_for, render_template, request

# Data Wrangling and Data Analysis
import pandas as pd
import numpy as np
import string
import re
import csv

# Model Building
import pickle

# Initialize Flask APP
app = Flask(__name__)

# Load Model
model = pickle.load(open('nlp_pipeline', 'rb'))

# Global Variables
Button = 0
Choice = 0
model_input = ""
model_output = ""

# Text Preprocessor
def clean_up(comments):
    # tokenize
    comments_tokens = [sent.lower().split() for sent in comments]

    # Remove punctuations and contextual Stopwords
    stop_context = ["article", "page",
                    "wikipedia", "talk", "articles", "pages"]
    stop_punct = list(string.punctuation)
    stop_final = stop_punct + stop_context + \
        ["...", "``", "''", " ", "" "====", "must"]

    def del_stop(sent):
        return [term for term in sent if term not in stop_final]
    cleaned_comments = [del_stop(sent) for sent in comments_tokens]
    cleaned_comments = [" ".join(word) for word in cleaned_comments]
    return(cleaned_comments)


@app.route('/')
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global Button 
    global model_input
    global model_output
    model_input = list(request.form.values())
    model_input = clean_up(model_input)
    model_output = model.predict(model_input)[0]
    Prediction = ""
    if model_output == 1:
        Prediction = "Comment is Negative ! \N{pensive face}"
    else:
        Prediction = "Comment is Positive ! \N{slightly smiling face}"

    if request.form.get('btn_predict'):
        Button = 1
    else:
        Button = 0
    return render_template('index.html', Prediction = Prediction, Button=Button)


@app.route('/retrain', methods=['POST'])
def retrain():

    # Retriving Global Variables
    global model_input
    global model_output
    global Button
    global Choice

    # Text Processing
    model_input = clean_up(model_input)

    # get user's button choice (right/wrong)
    button_type = request.form["choice"]

    # Changing the weights if prediction is wrong
    y_pred = np.array(0)
    if button_type == "wrong":
        if(model_output == 1):
            y_pred = np.array(0, ndmin=1)
        elif(model_output == 0):
            y_pred = np.array(1, ndmin=1)
        else:
            print("No Choices Recieved")
    else:
        y_pred = np.array(model_output, ndmin=1)

    # Vectorize the Text
    word_vec = model['Bow'].transform(model_input).toarray()
    word_vec = word_vec[0].reshape(1, -1)
    print("In retrain X ", word_vec)
    print("In retrain y ",y_pred)
    print("In retrain Type y ",type(y_pred))

    # Strengthen weights 
    model['model'].partial_fit(word_vec, y_pred)

    # Save trained model pickle
    pickle.dump(model, open('nlp_pipeline', 'wb'))

    # Saving the new Traning data
    fields = [model_input, model_output]

    # Writing the New Supervised Data Into CSV File
    with open(('user_teaching_data.csv'), 'a') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
    
    if (button_type == "right") or (button_type == "wrong"):
        Button = 0
        Choice = 1
    # return confirmation code for user
    return render_template('index.html', Choice=Choice)


if __name__ == '__main__':
    app.run(debug=True)
