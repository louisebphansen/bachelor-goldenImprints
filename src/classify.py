import argparse
import os 
import time
import datasets
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Input, InputLayer
from keras.models import Model
from sklearn.metrics import classification_report

# define argument parser
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str, help="name of train dataset")
    parser.add_argument('--test_data', type=str, help="name of test dataset")
    parser.add_argument('--val_data', type=str, help="name of validation data")
    parser.add_argument('--feature_col', type=str, help="name of column with class labels")
    parser.add_argument('--embedding_col', type=str, help="name of column containing the embeddings")
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for")
    parser.add_argument('--hidden_layer_size', type=int, help='size of the hidden layer in classification model')
    parser.add_argument('--batch_size', type=int)

    args = vars(parser.parse_args())
    
    return args

# load datasets

def load_dataset_from_dir(train_data, test_data, val_data):

    # define paths to train and test huggingface datasets
    path_to_train_ds = os.path.join('datasets', train_data)
    path_to_test_ds = os.path.join('datasets', test_data)
    path_to_val_ds = os.path.join("datasets", val_data)

    # load as huggingface datasets
    ds_train = datasets.load_from_disk(path_to_train_ds)
    ds_test = datasets.load_from_disk(path_to_test_ds)
    ds_val = datasets.load_from_disk(path_to_val_ds)

    return ds_train, ds_test, ds_val

# build classification model
def build_classfication_model(train_data, hidden_layer_size, feature_col, embedding_col):
    
    # save number of classes (to be used for the last layer of the model)
    num_classes = train_data.features[feature_col].num_classes

    # define input shape
    inp_size = len(train_data[0][embedding_col])
    inp = Input(shape=(inp_size,))

    # define shape of hidden layer
    hidden_layer = Dense(hidden_layer_size, activation='relu')(inp)
    
    # add classification layer
    classification_layer = Dense(num_classes, activation='softmax')(hidden_layer)

    # define model
    model = Model(inputs=inp, outputs=classification_layer)

    # compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy') # supposed to be the best for multiclass classification with non-one hot encoded labels?
    
    return model

def save_plot_history(H, epochs, name):
    '''
    Saves the validation and loss history plots of a fitted model in the 'out' folder.
    Code is borrowed and adapted from the Session 9 notebook of the Visual Analytics course @ Aarhus University, 2023.
    
    Arguments:
    - H: Saved history of a model fit
    - epochs: Number of epochs the model runs on
    - name: What the plot should be called
    
    Returns:
        None
    '''
    #plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join('plots', name))

# create model that classifies artist, genre etc. 
def fit_and_predict(train_data, test_data, val_data, hidden_layer_size, embedding_col, feature_col, batch_size, epochs):

    '''fit a compiled model on training data and predict on test dataset'''

    model = build_classfication_model(train_data, hidden_layer_size, feature_col, embedding_col)

    # convert to tensorflow datasets
    tf_ds_train = train_data.to_tf_dataset(
            columns=embedding_col, # the columns to be used as inputs to the model, X
            label_cols=feature_col, # columns containing class labels, y
            batch_size=batch_size,
            shuffle=True
            )
    
    tf_ds_test = test_data.to_tf_dataset(
            columns=embedding_col,
            label_cols=feature_col, 
            batch_size=batch_size,
            shuffle=False # for test data, set shuffle to false
            )
    
    tf_ds_val = val_data.to_tf_dataset(
            columns=embedding_col,
            label_cols=feature_col, 
            batch_size=batch_size,
            shuffle=False # ?
            )

    # define steps per epoch
    epoch_steps = len(train_data) // batch_size

    # fit model and save history
    H = model.fit(tf_ds_train, epochs = epochs, steps_per_epoch = epoch_steps, validation_data=tf_ds_val)

    # save history plot in "plots" folder
    #save_plot_history(H, epochs, f'{embedding_col}_{feature_col}_history.png')

    # predict on test data
    predictions = model.predict(tf_ds_test)

    # find class with the highest probability
    predicted_classes = np.argmax(predictions,axis=1)

    return predicted_classes

def save_classification_report(test_data, feature_col, embedding_col, predicted_classes):

    # save the class labels
    label_class = test_data.features[feature_col]

    # save the number of classes
    num_classes = test_data.features[feature_col].num_classes

    # map integer values to class label strings
    mapped_labels = {}

    for i in range(num_classes):
        mapped_labels[i] = label_class.int2str(i)
    
    labels = list(mapped_labels.values())

    # save classification report for y_true and y_pred
    report = classification_report(test_data[feature_col],
                            predicted_classes, target_names = labels)
    
    # save classification report
    out_path = os.path.join("classification_reports", f'{embedding_col}_{feature_col}_classification_report.txt')

    with open(out_path, 'w') as file:
                file.write(report)

def main():

    args = argument_parser()

    # load datasets
    ds_train, ds_test, ds_val = load_dataset_from_dir(args['train_data'], args['test_data'], args['val_data'])
 
    # start timer
    start_time = time.time()

    # fit model on train data and predict on test data
    predicted_classes = fit_and_predict(ds_train, ds_test, ds_val, args['hidden_layer_size'], args['embedding_col'], args['feature_col'], args['batch_size'], args['epochs'])

    end_time = time.time() - start_time

    # save time as txt
    with open(f"times/{args['embedding_col']}_{args['feature_col']}_training_duration.txt", "w") as f:
        f.write(str(end_time))

    # save classification report
    save_classification_report(test_ds_comp, args['feature_col'], args['embedding_col'], predicted_classes)

if __name__ == '__main__':
   main()