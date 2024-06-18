'''
This script classifies the WikiArt dataset based on the features extracted in the 'extract_features.py' script
Classification report and history plots are saved in the 'out' folder
'''
import argparse
import os
import time
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# define argument parser
def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, help='name/prefix of huggingface dataset to be used for training, testing and validation containing extracted embeddings. must be in the /datasets folder')
    parser.add_argument('--feature_col', type=str, help="name of column with class labels")
    parser.add_argument('--embedding_col', type=str, help="name of column containing the embeddings")
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for")
    parser.add_argument('--hidden_layer_size', type=int, help='size of the hidden layer in classification model')
    parser.add_argument('--batch_size', type=int)
    args = vars(parser.parse_args())
    return args

# load datasets
def load_data_from_dir(data_name):
    # define paths to train, test and validation datasets
    path_to_train_ds = os.path.join('datasets', f"{data_name}_train")
    path_to_test_ds = os.path.join('datasets', f"{data_name}_test")
    path_to_val_ds = os.path.join('datasets', f"{data_name}_val")
    ds_train = datasets.load_from_disk(path_to_train_ds).shuffle(seed=1848)
    ds_test = datasets.load_from_disk(path_to_test_ds)
    ds_val = datasets.load_from_disk(path_to_val_ds)
    return ds_train, ds_test, ds_val

# build classification model
class ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_classes):
        super(ClassificationModel, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_layer_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

def save_plot_history(history, epochs, name):
    '''saves the validation and loss history plots of a fitted model in the 'out' folder.'''
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), history['train_loss'], label="train_loss")
    plt.plot(np.arange(0, epochs), history['val_loss'], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), history['train_accuracy'], label="train_acc")
    plt.plot(np.arange(0, epochs), history['val_accuracy'], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join('out', 'plots', name))

def fit_and_predict(train_data, test_data, val_data, hidden_layer_size, embedding_col, feature_col, batch_size, epochs):
    '''fit a compiled model on training data and predict on test dataset'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = len(train_data[0][embedding_col])
    num_classes = train_data.features[feature_col].num_classes

    model = ClassificationModel(input_size, hidden_layer_size, num_classes).to(device)

    # turn images into single precision tensors
    train_embeddings = torch.tensor(np.vstack(train_data[embedding_col]), dtype=torch.float32)
    train_labels = torch.tensor(train_data[feature_col], dtype=torch.long)

    val_embeddings = torch.tensor(np.vstack(val_data[embedding_col]), dtype=torch.float32)
    val_labels = torch.tensor(val_data[feature_col], dtype=torch.long)

    test_embeddings = torch.tensor(np.vstack(test_data[embedding_col]), dtype=torch.float32)
    test_labels = torch.tensor(test_data[feature_col], dtype=torch.long)

    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)

    # moved shuffling to load_data_from_dir() to control the seed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # optimizer (Adam + sparse_categorical_crossentropy)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # training loop
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        # evaluation (validation) loop (runs every epoch)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    num_epochs = len(history['val_loss'])

    # save history plot in "plots" folder
    save_plot_history(history, num_epochs, f'{embedding_col}_{feature_col}_history.png')

    # predict on test data
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())

    predicted_classes = np.array(all_predictions)
    np.save(f'out/y_pred/{embedding_col}_{feature_col}_y_pred.npy', predicted_classes)

    return predicted_classes

def save_classification_report(test_data, feature_col, embedding_col, predicted_classes):
    label_class = test_data.features[feature_col]
    num_classes = test_data.features[feature_col].num_classes
    # map integer values to class label strings
    mapped_labels = {}
    for i in range(num_classes):
        mapped_labels[i] = label_class.int2str(i)
    labels = list(mapped_labels.values())
    # save classification report for y_true and y_pred
    report = classification_report(test_data[feature_col], predicted_classes, target_names=labels)
    out_path = os.path.join("out", "classification_reports", f'{embedding_col}_{feature_col}_classification_report.txt')
    with open(out_path, 'w') as file:
        file.write(report)

def main():
    args = argument_parser()
    ds_train, ds_test, ds_val = load_data_from_dir(args['data_name'])
    predicted_classes = fit_and_predict(ds_train, ds_test, ds_val, args['hidden_layer_size'], args['embedding_col'], args['feature_col'], args['batch_size'], args['epochs'])
    save_classification_report(ds_test, args['feature_col'], args['embedding_col'], predicted_classes)

if __name__ == '__main__':
    main()
