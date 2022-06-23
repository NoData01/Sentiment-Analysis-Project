# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:54:06 2022

@author: _K
"""

from tensorflow.keras.layers import LSTM,Dense,Dropout,Embedding
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
import matplotlib.pyplot as plt
import numpy as np


class ModelCreation():
    def __init__(self):
        pass
    
    def simple_lstm_layer(self, num_feature, num_class, vocab_size,
                          embedding_dim, drop_rate=0.3, num_node=128):
        
        model=Sequential()
        model.add(Input(shape=(num_feature))) #np.shape(X_train)[1] 
        model.add(Embedding(vocab_size,embedding_dim))
        model.add(SpatialDropout1D(0.4))
        model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
        model.add(Dropout(drop_rate))
        model.add(LSTM(num_node))
        model.add(Dropout(drop_rate))
        model.add(Dense(num_node,activation='relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(num_class,activation='softmax'))#output layer
        model.summary()
        
        return model


class HistHistory():
    def __init__(self):
        pass
    
    def plot_hist_loss(self, tr_loss, val_loss):
        plt.figure()
        plt.plot(tr_loss)
        plt.plot(val_loss)
        plt.legend('training_loss','validation_loss')
        plt.show()
    
    def plot_hist_acc(self, tr_acc, val_acc):
        plt.figure()
        plt.plot(tr_acc)
        plt.plot(val_acc)
        plt.legend('training_accuracy','validation_accuracy')
        plt.show()
        
        
        