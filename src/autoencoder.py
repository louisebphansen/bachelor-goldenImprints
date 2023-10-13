import datasets
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model

# define argparse arguments
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str, help= 'what data to use to train the model')
    parser.add_argument('--test_data', type=str, help='what data to use to test the model on')
    parser.add_argument('--hidden_layer_size', type=int, help='what the size of the hidden layer should be')
    parser.add_argument('--encoded_layer_size', type=int, help='what the size of the encoded layer should be')
    parser.add_argument('--reconstructions_col_name', type=str, help='what the name of the new column with the reconstructions should be')
    parser.add_argument('--new_dataset_name', type=str, help='what the name of the new dataset with the added reconstructions column should be')
    
    args = vars(parser.parse_args())
    
    return args

# define autoencoders

def build_autoencoder(inp_shape, hidden_layer_size, encoded_layer_size):

    input_img = tf.keras.layers.Input(shape=(inp_shape,), name='input')
    enc1 = tf.keras.layers.Dense(hidden_layer_size, activation = 'relu', name='encoder1')(input_img)
    
    embedding = tf.keras.layers.Dense(encoded_layer_size, activation = 'relu')(enc1)
    
    dec2 = tf.keras.layers.Dense(hidden_layer_size, activation = 'relu', name='decoder2')(embedding)
    decoded = tf.keras.layers.Dense(inp_shape, activation = 'relu', name='decoder3')(dec2)
    
    autoencoder = tf.keras.models.Model(inputs=input_img, outputs=decoded, name='AE'), tf.keras.models.Model(inputs=embedding, outputs=decoded, name='decoder')

    return autoencoder

    print(autoencoder.summary())

def train_and_predict(autoencoder, train_data, test_data, epochs):

    autoencoder.compile(loss='mse', optimizer='adam')
    
    autoencoder.fit(train_data, train_data, epochs=epochs)

    reconstructions = autoencoder.predict(test_data)

    return reconstructions 

def add_reconstructions_column(dataset, reconstructions, colname, new_dataset_name):

    # add embeddings to data
    new_dataset = dataset.add_column(colname, reconstructions)

    # save to datasets folder as new dataset
    new_dataset.save_to_disk(os.path.join('datasets', new_dataset_name))


def main():
    
    args = argument_parser()

    input_shape = len(args['train_data'][0][args['feature_col_name']])

    autoencoder = build_autoencoder(inp_shape, args['hidden_layer_size'], args['encoded_layer_size'])

    reconstructions = train_and_predict(autoencoder, args['train_data'], args['test_data'], args['epochs'])

    add_reconstructions_column(args['test_data'], reconstructions, args['reconstructions_col_name'], args['new_dataset_name'])

if __name__ == '__main__':
   main()
    


