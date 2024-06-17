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

    parser.add_argument('--pretrained_model', type=str, help= 'name of the pretrained model in timm to use')
    parser.add_argument('--data_name', type=str, help='name/prefix of huggingface dataset to be used for training, testing and validation. must be in the /datasets folder')
    parser.add_argument('--embedding_col_name', type=str, help='what the name of the new column containing the embeddings in the dataset should be')
    parser.add_argument('--new_data_name', type=str, help='what the train, test and validation datasets with the new embeddings columns should be saved as. datasets cant overwrite themselves, so give a new name. will be saved as three seperate train, test and val sets')

    args = vars(parser.parse_args())
    
    return args

def load_data_from_dir(data_name):

    # define paths to train, test and validation datasets
    path_to_train_ds = os.path.join('datasets', f"{data_name}_train")
    path_to_test_ds = os.path.join('datasets', f"{data_name}_test")
    path_to_val_ds = os.path.join('datasets', f"{data_name}_val")

    # load from disk
    ds_train = datasets.load_from_disk(path_to_train_ds)
    ds_test = datasets.load_from_disk(path_to_test_ds)
    ds_val = datasets.load_from_disk(path_to_val_ds)

    return ds_train, ds_test, ds_val

def save_preprocessing_info(model, model_name):

    # save preprocessing information from the pretrained model
    data_config = timm.data.resolve_model_data_config(model)

    # use this information to transform the data
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # save transformations as a .txt file
    transforms_str = str(transforms)

    with open(f'out/preprocessing/{model_name}_transforms.txt', 'w') as f:
        f.write(transforms_str)
    
    return transforms

def transform_and_extract(img, model, transforms):

    # apply transformations
    transformed_image = transforms(img).unsqueeze(0) # unsqueeze adds a dim so the shape is now (1, 3, img_size, img_size)

    # if running on GPU:
    if torch.cuda.is_available():
        transformed_image = transformed_image.cuda()
        model.to('cuda')
    
    # extract features
    features = model(transformed_image)

    # convert from tensor to list
    feature_list = features.tolist()

    # un-nest list
    feature_list_unnest = feature_list[0]

    return feature_list_unnest


def features_from_dataset(dataset, model, transforms):

    # initialize empty list
    embeddings = []

    # loop over each image in the dataset and extract feature embeddings
    for i in tqdm(range(len(dataset)), desc="Extracting features from images"):
        image = dataset[i]['image']
        feature = transform_and_extract(image, model, transforms)
        embeddings.append(feature)

    return embeddings

def add_feature_column(dataset, embeddings, colname, new_data_name):

    # add embeddings to data
    new_dataset = dataset.add_column(colname, embeddings)

    # save to datasets folder as new dataset
    new_dataset.save_to_disk(os.path.join('datasets', new_data_name))

def main():
    
    # parse arguments
    args = argument_parser()

    # check cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # initialize model
    model = timm.create_model(args['pretrained_model'], pretrained=True, num_classes=0)
    model.eval() # turn on evaluation mode
  
    # load huggingface datasets from desk
    train_ds, test_ds, val_ds = load_data_from_dir(args['data_name'])

    # save the preprocessing steps applied to the data
    transforms = save_preprocessing_info(model, args['pretrained_model'])

    # start timer
    start_time = time.time()

    # extract features for training dataset
    features_train = features_from_dataset(train_ds, model, transforms)
    end_time = time.time() - start_time

    # save time as txt
    model_name = args['pretrained_model']
    with open(f"out/times/{args['pretrained_model']}_feature_extraction.txt", "w") as f:
        f.write(str(end_time))

    # extract features for test and validation datasets
    features_test = features_from_dataset(test_ds, model, transforms)
    features_val = features_from_dataset(val_ds, model, transforms)

    # save new train and test-datasets with added embeddings columns
    add_feature_column(train_ds, features_train, args['embedding_col_name'], f"{args['new_data_name']}_train")
    add_feature_column(test_ds, features_test, args['embedding_col_name'], f"{args['new_data_name']}_test")
    add_feature_column(val_ds, features_val, args['embedding_col_name'], f"{args['new_data_name']}_val")

if __name__ == '__main__':
   main()