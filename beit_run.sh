source env/bin/activate

#python3 src/extract_features_gpu.py --pretrained_model beit_large_patch16_512.in22k_ft_in22k_in1k --data_name WikiArt --embedding_col_name beit_large --new_data_name beit_large

python3 src/classify.py --data_name beit_large --feature_col genre --embedding_col beit_large --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python3 src/classify.py --data_name beit_large --feature_col style --embedding_col beit_large --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python3 src/classify.py --data_name beit_large --feature_col artist --embedding_col beit_large --epochs 20 --hidden_layer_size 3000 --batch_size 32 