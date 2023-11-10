''' 
This script can be used to visually 'check' the embeddings extracted by a given model. Calculates similar images with a 
kNN algorithm and plots the 5 closest images to a target image.

'''

import datasets
import numpy as np
import matplotlib.pyplot as plt
import argparse
from numpy.linalg import norm
from functools import partial
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str)
    parser.add_argument('--target_image', type=int)
    parser.add_argument('--feature_list', type=str)
    parser.add_argument('--plot_name', type=str)

    args = vars(parser.parse_args())
    
    return args

def find_neighbors(feature_list, target_image):
    '''
    Calculate nearest neighbors to a target image using a feature list
    Saves a .csv file from a Pandas dataframe containing the 5 closest images

    Arguments:
    - feature_list: list of extracted features for each image
    - filenames: list of filenames for data directory
    - target_image: name of target image
    - out_folder: where to save the output csv

    Returns:
    - A Pandas series containing the names of the 5 nearest neighbors.
    '''

    # initialize K-nearest neighbors algorithm
    neighbors = NearestNeighbors(n_neighbors=10, 
                            algorithm='brute',
                            metric='cosine').fit(feature_list)

    # find the index of target image in filenames list
    

    # save the indices and distances of the neighbors to the target image
    distances, indices = neighbors.kneighbors([feature_list[target_image]])

    # initialize empty lists
    idxs = []
    dist = []
    
        # save the 5 closest images' indices and distances
    for i in range(1,6):
        idxs.append(indices[0][i])
        dist.append(distances[0][i])
    

    # create dataframe
    data = pd.DataFrame({
                        "distance_score" : pd.Series(dist),
                        'index': pd.Series(idxs)})
    
    # save as csv
    #data.to_csv(os.path.join("out", out_folder, f"{target_image}.csv"))
    
    # return filenames as a pandas series to be used in the plotting function
    return data

def show_plot(names, target_image, dataset, plot_name):
    '''
    Plot target image next to the five closest images

    Arguments:
    - path: path to folder where image data is stored
    - names: the filenames of the 5 closest images. Must be a pandas series
    - target_image: target input image
    - folder: specifies in what folder in output to save the plot to
    
    Returns:
    None
    '''
    
    # arrange plots
    f, axarr = plt.subplots(2, 3)
    
    # print target image
    axarr[0,0].imshow(dataset[target_image]['image'])
    axarr[0,0].title.set_text('Target Image')

    # plot 5 most similar next to it
    axarr[0,1].imshow(dataset[names[0]]['image'])
    axarr[0,2].imshow(dataset[names[1]]['image'])
    axarr[1,0].imshow(dataset[names[2]]['image'])
    axarr[1,1].imshow(dataset[names[3]]['image'])
    axarr[1,2].imshow(dataset[names[4]]['image'])
    
    # remove axes from plot
    for ax in f.axes:
        ax.axison = False

    #plt.show()

    plt.savefig(os.path.join('plots', plot_name))

def plot_neighbors(feature_list, target_image, dataset, plot_name):

    # find closest images and save in a df
    data = find_neighbors(feature_list, target_image)

    # save the indices of the closest images
    indices = data['index'].tolist()

    # plot them
    show_plot(indices, target_image, dataset, plot_name)

def main():

    args = argument_parser()

    path_to_train_ds = os.path.join('datasets', args['data'])

    ds = datasets.load_from_disk(path_to_train_ds)

    plot_neighbors(ds[args['feature_list']], args['target_image'], ds, args['plot_name'])

if __name__ == '__main__':
   main()



