#!/usr/bin/env python
# coding: utf-8

# In[57]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random
import os

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')  # Add this line
nltk.download('punkt_tab')

# Load the model
model_path = 'D:/target/ml/ChatBot/chatbot_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)

# Load intents and processed data
intents = json.loads(open(r"D:\target\ml\ChatBot\intents.json", encoding="utf8").read())
words = pickle.load(open(r"D:\target\ml\ChatBot\words.pkl", 'rb'))
classes = pickle.load(open(r"D:\target\ml\ChatBot\classes.pkl", 'rb'))


# In[58]:


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# In[59]:


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


# In[60]:


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.10
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# In[61]:

def getResponse(ints, intents_json):
    if not ints:  # Check if the list is empty
        return "I'm sorry, I didn't understand that."
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result



# In[62]:


def chatbot_response(msg):
    if not msg.strip():
        return "Please enter a valid question."
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res



# In[64]:


from flask import Flask, jsonify


app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def hello():
    return jsonify({"key" : "home page value"})

#function to replace '+' character with ' ' spaces
def decrypt(msg):
    
    string = msg
    
    #converting back '+' character back into ' ' spaces
    #new_string is the normal message with spaces that was sent by the user
    new_string = string.replace("+", " ")
    
    return new_string

#here we will send a string from the client and the server will return another
#string with som modification
#creating a url dynamically
@app.route('/home/<name>') 
def hello_name(name):
    
    #dec_msg is the real question asked by the user
    dec_msg = decrypt(name)
    
    #get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg)
    
    #creating a json object
    json_obj = jsonify({"top" : {"res" : response}})
    
    return json_obj



if __name__ == '__main__':
    app.run(debug=True , threaded = True)


# In[ ]:





# In[ ]:




