import argparse
import os
import time
import timm
import torch
import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class CustomImageDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        transformed_image = self.transforms(image)
        return transformed_image


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model', type=str, help= 'name of the pretrained model in timm to use')
    parser.add_argument('--data_name', type=str, help='name/prefix of huggingface dataset to be used for training, testing and validation. must be in the /datasets folder')
    parser.add_argument('--embedding_col_name', type=str, help='what the name of the new column containing the embeddings in the dataset should be')
    parser.add_argument('--new_data_name', type=str, help='what the train, test and validation datasets with the new embeddings columns should be saved as. datasets cant overwrite themselves, so give a new name. will be saved as three seperate train, test and val sets')

    args = vars(parser.parse_args())
    return args

def load_data_from_dir(data_name):
    path_to_train_ds = os.path.join('datasets', f"{data_name}_train")
    path_to_test_ds = os.path.join('datasets', f"{data_name}_test")
    path_to_val_ds = os.path.join('datasets', f"{data_name}_val")

    ds_train = datasets.load_from_disk(path_to_train_ds)
    ds_test = datasets.load_from_disk(path_to_test_ds)
    ds_val = datasets.load_from_disk(path_to_val_ds)

    return ds_train, ds_test, ds_val

def save_preprocessing_info(model, model_name):
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    transforms_str = str(transforms)

    with open(f'out/preprocessing/{model_name}_transforms.txt', 'w') as f:
        f.write(transforms_str)

    return transforms

def features_from_dataset(dataset, model, transforms, device, batch_size=32):
    embeddings = []
    custom_dataset = CustomImageDataset(dataset, transforms)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features from images"):
            batch = batch.to(device)
            features = model(batch)
            embeddings.extend(features.cpu().tolist())

    return embeddings

def add_feature_column(dataset, embeddings, colname, new_data_name):
    new_dataset = dataset.add_column(colname, embeddings)
    new_dataset.save_to_disk(os.path.join('datasets', new_data_name))

def main():
    args = argument_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = timm.create_model(args['pretrained_model'], pretrained=True, num_classes=0).to(device)
    model.eval()

    train_ds, test_ds, val_ds = load_data_from_dir(args['data_name'])
    transforms = save_preprocessing_info(model, args['pretrained_model'])

    start_time = time.time()
    features_train = features_from_dataset(train_ds, model, transforms, device)
    end_time = time.time() - start_time

    model_name = args['pretrained_model']
    with open(f"out/times/{args['pretrained_model']}_feature_extraction.txt", "w") as f:
        f.write(str(end_time))

    features_test = features_from_dataset(test_ds, model, transforms, device)
    features_val = features_from_dataset(val_ds, model, transforms, device)

    add_feature_column(train_ds, features_train, args['embedding_col_name'], f"{args['new_data_name']}_train")
    add_feature_column(test_ds, features_test, args['embedding_col_name'], f"{args['new_data_name']}_test")
    add_feature_column(val_ds, features_val, args['embedding_col_name'], f"{args['new_data_name']}_val")

if __name__ == '__main__':
    main()
