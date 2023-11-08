source env/bin/activate # activate virtual environment

# run scripts
#python3 src/load_hf_dataset.py --huggingface_dataset huggan/wikiart --name test --seed 2830
#python3 src/extract_features.py --pretrained_model eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --train_data wiki6k_train --test_data wiki6k_test --feature_col_name eva02_base --new_traindata_name wiki6k_train_eva02 --new_testdata_name wiki6k_test_eva02
python3 src/extract_features.py --pretrained_model eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --data_name test --embedding_col_name eva02_base --new_data_name test
#python3 src/classify.py --train_data wiki6k_train_eva02 --test_data wiki6k_test_eva02 --val_data wiki6k_val_eva02 --feature_col genre --embedding_col eva02_base --epochs 2 --hidden_layer_size 350 --batch_size 32 

deactivate # deactivate virtual environment again