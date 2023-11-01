import argparse
import os 
import time
import timm
import torch
import datasets
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model', type=str, help= 'name of the pretrained model to use')
    
    args = vars(parser.parse_args())
    
    return args

# load dataset with features columns

def load_data_from_desk(train_data, test_data):

    path_to_train_ds = os.path.join('datasets', train_data)
    path_to_test_ds = os.path.join('datasets', test_data)

    ds_train = datasets.load_from_disk(path_to_train_ds)
    ds_test = datasets.load_from_disk(path_to_test_ds)

    return ds_train, ds_test

# compress the data using a simple encoder

def compress_data(ds_train, feature_col, compressed_size, epochs):
    # change this !
    inp_shape = len(ds_train[0][feature_col])
    inp = Input(shape=(inp_shape,))
    compressed = Dense(compressed_size, activation = 'relu')(inp) # problem at bruge relu her ??? 
    out = Dense(inp_shape, activation = 'relu')(compressed)
    model = Model(inputs=inp, outputs = out)

    comp_model.compile(optimizer='adam', loss='mse')
    comp_model.fit(ds_train[feature_col], ds_train[feature_col], epochs=epochs)
    encoder = Model(comp_model.input, comp_model.layers[-2].output)
    compressions = encoder.predict(ds_train[feature_col])

    return model

def classfication_model(data, inp_size, hidden_layer_size, feature):
    
    len(set(data[feature]))

    inp = Input(shape=(inp_size,))
    hidden_layer = Dense(hidden_layer_size, activation='relu')(inp)
    classification_layer = Dense()



# create neural network that classifies artist, genre etc. 
def classify(train_data, embedding_col, inp_size, hidden_layer_size):

    embeddings = train_data[feature_col]

    inp = Input(shape=(inp_size,))
    hidden_layer = Dense(hidden_layer_size, activation='relu')(inp)
    classification_layer = Dense()








# save classification report for each model

if __name__ == '__main__':
   main()