import argparse
import os 
import time
import timm
import torch
import datasets
from tqdm import tqdm

# define argument parser
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model', type=str, help= 'name of the pretrained model to use')
    parser.add_argument('--train_data', type=str, help='name of huggingface dataset to be used for training. must be in the /datasets folder')
    parser.add_argument('--test_data', type=str, help='name of huggingface dataset to be used for testing. must be in the /datasets folder')
    parser.add_argument('--feature_col_name', type=str, help='what the name of the new column in the dataset should be')
    parser.add_argument('--new_traindata_name', type=str, help='what the train dataset with the new embeddings column shoukd be saved as. datasets cant overwrite themselves, so give a new name')
    parser.add_argument('--new_testdata_name', type=str, help='what the test dataset with the new embeddings column shoukd be saved as. datasets cant overwrite themselves, so give a new name')
    
    args = vars(parser.parse_args())
    
    return args

def load_data_from_desk(train_data, test_data):

    path_to_train_ds = os.path.join('datasets', train_data)
    path_to_test_ds = os.path.join('datasets', test_data)

    ds_train = datasets.load_from_disk(path_to_train_ds)
    ds_test = datasets.load_from_disk(path_to_test_ds)

    return ds_train, ds_test

def save_preprocessing_info(model, model_name):

    data_config = timm.data.resolve_model_data_config(model)

    transforms = timm.data.create_transform(**data_config, is_training=False)

    # save transformations as a .txt file
    transforms_str = str(transforms)

    with open(f'preprocessing/{model_name}_transforms.txt', 'w') as f:
        f.write(transforms_str)
    
    return transforms

def transform_and_extract(img, model, transforms):

    # apply transformations, convert to tensor and extract features
    features = model(transforms(img).unsqueeze(0)) # unsqueeze adds a dim so the shape is now (1, 3, img_size, img_size)

    # convert from tensor to list
    feature_list = features.tolist()

    # un-nest list
    feature_list_unnest = feature_list[0]

    return feature_list_unnest


def features_from_dataset(dataset, model, transforms):

    # initialize empty list
    embeddings = []

    # loop over each image in the dataset
    for i in tqdm(range(len(dataset)), desc="Extracting features from images"):
        image = dataset[i]['image']
        # extract feature embeddings
        feature = transform_and_extract(image, model, transforms)
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
    model.eval()
  
    # load huggingface datasets from desk
    train_ds, test_ds = load_data_from_desk(args['train_data'], args['test_data'])

    # save the preprocessing steps applied to the data
    transforms = save_preprocessing_info(model, args['pretrained_model'])

    # start timer
    start_time = time.time()

    # extract features for training dataset
    features_train = features_from_dataset(train_ds, model, transforms)
    end_time = time.time() - start_time

    # save time as txt
    model_name = args['pretrained_model']
    with open(f'times/{model_name}_features_duration.txt', 'w') as f:
        f.write(str(end_time))

    # extract features for test dataset
    features_test =  features_from_dataset(test_ds, model, transforms)

    # save new train and test-datasets with added embeddings columns
    add_feature_column(train_ds, features_train, args['feature_col_name'], args['new_traindata_name'])
    add_feature_column(test_ds, features_test, args['feature_col_name'], args['new_testdata_name'])

if __name__ == '__main__':
   main()