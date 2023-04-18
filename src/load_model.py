from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# Define a learning rate schedule

    
def create_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dim, activation='relu', kernel_initializer=GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu', kernel_initializer=GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_initializer=GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_initializer=GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_initializer=GlorotUniform()))
    model.add(Dense(1, activation='sigmoid'))
    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def load_my_model(input_dim):
    model = create_mlp_model(input_dim)
    model.load_weights("quantum_mlp_model.h5")
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model = compile_model(model)
    return model

input_dim = 28  
model = load_my_model(input_dim)