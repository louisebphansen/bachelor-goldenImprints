source env/bin/activate # activate virtual environment

# run scripts
#python3 src/load_hf_dataset.py --huggingface_dataset huggan/wikiart --train_ds_name testy_train --test_ds_name testy_test

python3 src/extract_features.py --pretrained_model resnet50 --train_data testy_train --test_data testy_test --feature_col_name resnet_features --new_traindata_name testy_train --new_testdata_name testy_test

deactivate # deactivate virtual environment again