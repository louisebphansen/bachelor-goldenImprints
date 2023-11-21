source env/bin/activate # activate virtual environment

# run scripts
#python3 src/load_hf_dataset.py --huggingface_dataset huggan/wikiart --name WikiArt --seed 2830
#python3 src/extract_features.py --pretrained_model eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --data_name WikiArt --embedding_col_name eva02_base --new_data_name eva02_base
#python3 src/classify.py --data_name eva02_base --feature_col genre --embedding_col eva02_base --epochs 15 --hidden_layer_size 700 --batch_size 32 
python3 src/classify.py --data_name eva02_base --feature_col genre --embedding_col eva02_base --epochs 15 --hidden_layer_size 700 --batch_size 32 
#python3 src/classify_test.py --data_name eva02_base --feature_col style --embedding_col eva02_base --epochs 15 --hidden_layer_size 700 --second_layer_size 200 --batch_size 32

deactivate # deactivate virtual environment again