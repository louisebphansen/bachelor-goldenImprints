'''
This script loads a dataset from the Huggingface Hub and splits it into train, test and validation sets. 
The datasets are saved in the 'datasets' folder in the main directory

'''
import datasets
from functools import partial
import argparse 
import os 

# define argument parser

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--huggingface_dataset', type=str, help='path to dataset on huggingface, e.g., huggan/wikiart')
    parser.add_argument('--name', type=str, help='what to call the new datasets')
    parser.add_argument('--seed', type=int, help='seed for the train/test split')
    
    args = vars(parser.parse_args())
    
    return args


def load_iter_hf_data(dataset_name):

    '''
    Load a dataset from Huggingface Hub and save locally

    Args:
        - dataset_name: name of dataset from Hugginface Hub
    '''

    # load dataset from the hub
    hf_data = datasets.load_dataset(dataset_name, split='train', streaming=True)

    # convert dataset to iterable generator
    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    # convert to dataset to be saved locally
    ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, hf_data), features=hf_data.features)

    return ds

def split_data(ds, name, seed):
    '''
    Split data into train, test and validation splits

    Args:
        - ds: saved huggingface dataset
        - name: prefix of saved dataset splits (e.g., 'WikiArt' will save datasets WikiArt_train, WikiArt_test etc.)
        - seed: set seed to ensure reproducability
    '''

    # split data into train and test
    ds_split = ds.train_test_split(test_size=0.2, seed=seed)
    ds_train = ds_split['train']
    ds_test = ds_split['test']

    # split test data into test and validation
    ds_test_split = ds_test.train_test_split(test_size=0.5, seed=seed)
    ds_val = ds_test_split['train']
    ds_test = ds_test_split['test']

    # save the datasets to disk with the same prefix
    ds_train.save_to_disk(os.path.join('datasets', f"{name}_train"))
    ds_test.save_to_disk(os.path.join('datasets', f"{name}_test"))
    ds_val.save_to_disk(os.path.join('datasets', f"{name}_val"))

def main():
    
    args = argument_parser()

    ds = load_iter_hf_data(args['huggingface_dataset'])

    split_data(ds, args['name'], args['seed'])

if __name__ == '__main__':
   main()