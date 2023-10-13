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

    args = vars(parser.parse_args())
    
    return args

# define autoencoders

def build_autoencoder(inp_shape, hidden_layer_size, latent_shape):
    #n_layers = 4
    input_img = tf.keras.layers.Input(shape=(inp_shape,), name='input')

    enc1 = tf.keras.layers.Dense(hidden_layer_size, activation = 'relu', name='encoder1')(input_img)
    
    embedding = tf.keras.layers.Dense(latent_shape, activation = 'relu')(enc1)
    
    dec2 = tf.keras.layers.Dense(hidden_layer_size, activation = 'relu', name='decoder2')(embedding)

    decoded = tf.keras.layers.Dense(inp_shape, activation = 'relu', name='decoder3')(dec2)

    autoencoder = tf.keras.models.Model(inputs=input_img, outputs=decoded, name='AE'), tf.keras.models.Model(inputs=embedding, outputs=decoded, name='decoder')

    return autoencoder


def main():
    
    args = argument_parser()

    input_shape = len(args['train_data'][0]args['feature_col_name'])

    small_autoencoder = build_autoencoder(inp_shape, hidden_layer_size, latent_shape)
