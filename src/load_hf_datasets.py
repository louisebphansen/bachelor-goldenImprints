'''
This script loads a dataset from the Huggingface Hub, splits it into test and train and saves in a folder in the directory

'''

import datasets # huggingface datasets package
from functools import partial
import argparse 
import os 

# define argument parser

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--huggingface_dataset', type=str, help='path to dataset on huggingface, e.g., huggan/wikiart')
    parser.add_argument('--train_ds_name', type=str, help= "what the name of the train data should be")
    parser.add_argument('--test_ds_name', type=str, help='what the name of the test dataset should be')
    
    args = vars(parser.parse_args())
    
    return args


def load_iter_hf_data(dataset_name):

    hf_data = datasets.load_dataset(dataset_name, split='train', streaming=True)

    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, hf_data), features=hf_data.features)

    return ds

def split_data(ds, train_ds_name, test_ds_name):

    ds_split = ds.train_test_split(test_size=test_size, seed=seed)
    ds_train = ds_split['train']
    ds_test = ds_split['test']

    ds_train.save_to_disk(os.path.join('datasets', train_ds_name))
    ds_test.save_to_disk(os.path.join('datasets', test_ds_name))

def main():
    
    args = argument_parser()

    ds = load_iter_hf_data(args['huggingface_dataset'])

    split_data(ds, args['train_ds_name'], args['test_ds_name'])

if __name__ == '__main__':
   main()