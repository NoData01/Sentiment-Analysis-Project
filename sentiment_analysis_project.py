# -*- coding: utf-8 -*-
"""Sentiment_Analysis.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kSHwgwDgqDU6HsioMVHrJ4KriKDf5VYV
"""

# Edited on 23 June 2022

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from module_sentiment import ModelCreation,HistHistory

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import pickle
import json
import os 


#%% Statics
CSV_URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-\
2019-NLP-Tutorial/master/bbc-text.csv'
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'Segmentation_logs', log_dir)

# EDA 
#%% Step1 - Load Data
df = pd.read_csv(CSV_URL)

#%% Step2 - Data Inspection/Visualization
df.head(10)
df.tail(10)
df.info()
df.describe()

df['category'].unique() # to get the unique target
df['text'][1]           # a business category for text[1]
df['category'][1] 


df.duplicated().sum()   # There is 99 duplicated text
df[df.duplicated()]

#%% Step3 - Data Cleaning
# Remove the duplicated data
df = df.drop_duplicates()
df.duplicated().sum()   # Ensure all duplicated have been removed

text = df['text'].values         # features of X
category = df['category'].values # target, y

#%% Step4 - Features Selection
# Nothing to select

#%% Step5 - Data Preprocessing
#           1) Convert into lower case (no upper case been detected in text)

#           2) Tokenization

vocab_size = 10000 # 10000 words will get characterized
oov_token = 'OOV'

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text)       # Learning all the words 
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(text) # To convert into numbers

#           3)Padding & Trunctation

length_of_text= [len(i) for i in train_sequences]    # List comprehension
print(np.median(length_of_text)) # to get the number of max length for padding

max_len = 340

padded_text = pad_sequences(train_sequences,maxlen = max_len,
                             padding='post',
                             truncating='post')

#           4)OneHotEncoding for the Target
ohe = OneHotEncoder(sparse=False)

with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)

category = ohe.fit_transform(np.expand_dims(category,axis=-1))

#           5)Train test split
X_train,X_test,y_train,y_test = train_test_split(padded_text,
                                                 category,
                                                 test_size=0.3,
                                                 random_state=123)

X_train = np.expand_dims(X_train,axis=-1) 
X_test = np.expand_dims(X_test,axis=-1)

#%% Model development

embedding_dim = 64
num_feature = np.shape(X_train)[1]
num_class = 5 

mc = ModelCreation()
model = mc.simple_lstm_layer(num_feature, num_class, vocab_size,
                             embedding_dim, drop_rate=0.3, num_node=128)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics='acc')

plot_model(model, to_file='model_plot.png', show_shapes=True, 
           show_layer_names=True)

# Callbacks
tensorboard_callback = TensorBoard(log_dir=LOG_FOLDER_PATH)

# Model training
hist = model.fit(X_train,y_train, validation_data=(X_test,y_test),
                 epochs=50, batch_size=128, callbacks=[tensorboard_callback])

# Plotting the graph

hist.history.keys()

training_loss = hist.history['loss']
validation_loss = hist.history['val_loss']

training_accuracy = hist.history['acc']
validation_accuracy = hist.history['val_acc']

ht = HistHistory()
ht.plot_hist_loss(training_loss,validation_loss)
ht.plot_hist_acc(training_accuracy,validation_accuracy)

#%% Model evaluation

results = model.evaluate(X_test,y_test)
print(results)
y_true = np.argmax(y_test,axis=1)
y_pred = np.argmax(model.predict(X_test),axis=1)

cm = confusion_matrix(y_true,y_pred)
cr = classification_report(y_true,y_pred)
print(cm)
print(cr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%% Model saving
model.save(MODEL_SAVE_PATH)

token_json = tokenizer.to_json()

with open(TOKENIZER_PATH,'w') as file:
  json.dump(token_json,file)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir Segmentation_logs