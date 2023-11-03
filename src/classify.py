import argparse
import os 
import time
import datasets
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Model
from sklearn.metrics import classification_report

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model', type=str, help= 'name of the pretrained model to use')
    parser.add_argument('--train_data', type=str, help="name of train dataset")
    parser.add_argument('--test_data', type=str, help="name of test dataset")
    parser.add_argument('--feature_col', type=str, help="name of column with class labels")
    parser.add_argument('--embedding_col', type=str, help="name of column containing the embeddings")
    parser.add_argument('--epochs', type=int, help="how many epochs to run the models for")
    parser.add_argument('--compressed_size', type=int, help="size of the final, compressed embedding vector")
    parser.add_argument('--hidden_layer_size', type=int, help='size of the hidden layer in classification model')
    parser.add_argument('--batch_size', type=int)

    args = vars(parser.parse_args())
    
    return args

# load dataset with features columns

def load_dataset_from_dir(train_data, test_data):

    # define paths to train and test huggingface datasets
    path_to_train_ds = os.path.join('datasets', train_data)
    path_to_test_ds = os.path.join('datasets', test_data)

    # load as huggingface datasets
    ds_train = datasets.load_from_disk(path_to_train_ds)
    ds_test = datasets.load_from_disk(path_to_test_ds)

    return ds_train, ds_test

# compress the data using a simple encoder
def compress_data(ds, embedding_col, compressed_size, epochs):
    # change this !
    inp_shape = len(ds[0][embedding_col])
    inp = Input(shape=(inp_shape,))
    compressed = Dense(compressed_size, activation = 'relu')(inp) # problem at bruge relu her ??? 
    out = Dense(inp_shape, activation = 'relu')(compressed)
    comp_model = Model(inputs=inp, outputs = out)

    comp_model.compile(optimizer='adam', loss='mse')
    comp_model.fit(ds[embedding_col], ds[embedding_col], epochs=epochs)
    encoder = Model(comp_model.input, comp_model.layers[-2].output)
    compressions = encoder.predict(ds[embedding_col])

    compressed_list = compressions.tolist()
    new_ds = ds.add_column(f'{embedding_col}_compressed', compressed_list)

    return new_ds

# build classification model
def build_classfication_model(train_data, hidden_layer_size, feature_col, embedding_col):
    
    # save number of classes
    num_classes = train_data.features[feature_col].num_classes
    inp_size = len(train_data[0][embedding_col])

    # define input shape
    inp = Input(shape=(inp_size,))

    # define shape of hidden layer
    hidden_layer = Dense(hidden_layer_size, activation='relu')(inp)
    
    # add classification layer with softmax activation
    classification_layer = Dense(num_classes, activation='softmax')(hidden_layer)

    # define model
    model = Model(inputs=inp, outputs=classification_layer)

    # compile model - maybe change loss function?
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    return model

# create neural network that classifies artist, genre etc. 
def fit_and_predict(model, train_data, test_data, embedding_col, feature_col, batch_size, epochs):

    compress_data = compress_data(train_data, embedding_col, compressed_size, epochs)

    model = build_classfication_model(train_data, hidden_layer_size, feature_col, embedding_col):

    '''fit a compiled model on training data and predict on test dataset'''

    tf_ds_train = train_data.to_tf_dataset(
            columns=embedding_col, # the columns to be used as inputs to the model, X
            label_cols=feature_col, # columns containing class labels, y
            batch_size=batch_size,
            shuffle=True
            )
    
    tf_ds_test = test_data.to_tf_dataset(
            columns=embedding_col, # the columns to be used as inputs to the model, X
            label_cols=feature_col, # columns containing class labels, y
            batch_size=batch_size,
            shuffle=False
            )

    epoch_steps = len(train_data) // batch_size

    H = model.fit(tf_ds_train, epochs = epochs, steps_per_epoch = epoch_steps)

    # save history plot? 

    # predict on test data
    predictions = model.predict(tf_ds_test)

    # find class with the highest probability
    predicted_classes = np.argmax(predictions,axis=1)

    return predicted_classes

def save_classification_report(test_data, feature_col, embedding_col, predicted_classes):

    label_class = test_data.features[feature_col]
    num_classes = test_data.features[feature_col].num_classes

    mapped_labels = {}

    for i in range(num_classes):
        mapped_labels[i] = label_class.int2str(i)
    
    report = classification_report(test_data[embedding_col],
                            predicted_classes, target_names = mapped_labels)
    

    out_path = os.path.join("classification_reports", 'classification_report.txt')

    with open(out_path, 'w') as file:
                file.write(report)

def main():

    args = argument_parser()

    # load datasets
    ds_train, ds_test = load_dataset_from_dir(args['train_data'], args['test_data'])

    # compress the size of the embedding, so all models have the same input size in the classifier
    new_train_ds = compress_data(ds_train, args['embedding_col'], args['compressed_size'], args['epochs'])
    new_test_ds =  compress_data(ds_test, args['embedding_col'], args['compressed_size'], args['epochs'])

    # create model (bad function, find a better way to do it? the inputs are not used properly)
    model = build_classfication_model(train_data=new_train_ds, inp_size=args['compressed_size'], hidden_layer_size=args['hidden_layer_size'], feature=args['feature_col'], embedding_col=args['embedding_col'])

    # fit model on train data and predict on test data
    predicted_classes = fit_and_predict(model, new_train_ds, new_test_ds, f'{args['embedding_col']}_compressed', args['feature_col'], args['batch_size'], args['epochs'])

    save_classification_report(new_test_ds, args['feature_col'], args['embedding_col'], predicted_classes)

if __name__ == '__main__':
   main()