# import
import argparse
import os 
import timm
import torch
import datasets
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array

# define argument parser
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model', type=str, help= 'what type of pretrained model to use')
    parser.add_argument('--train_data', type=str, help='name of huggingface dataset to be used for training. must be in the /datasets folder')
    parser.add_argument('--test_data', type=str, help='name of huggingface dataset to be used for testing. must be in the /datasets folder')
    parser.add_argument('--feature_col_name', type=str, help='what the name of the new column in the dataset should be')
    parser.add_argument('--new_traindata_name', type=str, help='what the train dataset with the new embeddings column shoukd be saved as')
    parser.add_argument('--new_testdata_name', type=str, help='what the test dataset with the new embeddings column shoukd be saved as')
    
    args = vars(parser.parse_args())
    
    return args


def feature_extraction(img, img_size, chosen_model):
    
    # resize image to fit with model
    img = img.resize((img_size, img_size))
    # convert to np.array
    img_array = img_to_array(img)
    # expand dimensions (n_samples, 3, img_size, img_size)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # transpose array to fit PyTorch's format
    array_transposed = np.transpose(expanded_img_array, (0, 3, 1, 2))
    # normalize
    preprocessed_img = (array_transposed/255.0)
    # convert to torch
    inp = torch.from_numpy(preprocessed_img)
    # extract features from model
    feature = chosen_model(inp)
    # convert from torch to list
    feature_list = feature.tolist()
    # un-nest list
    feature_list = feature_list[0]

    return feature_list


def features_from_dataset(dataset, img_size, model):

    # initialize empty list
    embeddings = []

    # loop over each image in the dataset
    for i in tqdm(range(len(dataset)), desc=f"Extracting features from {dataset}"):
        image = dataset[i]['image']
        # extract feature embeddings
        feature = feature_extraction(image, img_size, model)
        embeddings.append(feature)

    return embeddings

def add_feature_column(dataset, embeddings, colname, new_dataset_name):

    # add embeddings to data
    new_dataset = dataset.add_column(colname, embeddings)

    # save to datasets folder as new dataset
    new_dataset.save_to_disk(os.path.join('datasets', new_dataset_name))

def main():
    
    # parse arguments
    args = argument_parser()

    # initialize model
    model = timm.create_model(args['pretrained_model'], pretrained=True, num_classes=0)
    
    # save input image size from pretrained model
    img_size = model.default_cfg['input_size'][0]

    path_to_train_ds = os.path.join('datasets', args['train_data'])
    path_to_test_ds = os.path.join('datasets', args['test_data'])
    
    # extract features for training dataset
    features_train = features_from_dataset(path_to_train_ds, img_size, model)
    
    # extract features for test dataset
    features_test =  features_from_dataset(path_to_test_ds, img_size, model)

    # save new train and test-datasets with added embeddings columns
    add_feature_column(path_to_test_ds, features_train, args['feature_col_name'], args['new_traindata_name'])
    add_feature_column(path_to_test_ds, features_test, args['feature_col_name'], args['new_testdata_name'])

if __name__ == '__main__':
   main()