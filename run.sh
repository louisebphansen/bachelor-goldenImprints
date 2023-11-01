source env/bin/activate # activate virtual environment

# run scripts
#python3 src/load_hf_dataset.py --huggingface_dataset huggan/wikiart --train_ds_name test_train --test_ds_name test_test --seed 2830

python3 src/extract_features.py --pretrained_model vgg16 --train_data test_train --test_data test_test --feature_col_name vgg16 --new_traindata_name test_train_lol --new_testdata_name test_test_lol

deactivate # deactivate virtual environment again