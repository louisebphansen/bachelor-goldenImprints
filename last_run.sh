source env/bin/activate

#python3 src/extract_features.py --pretrained_model convnext_xxlarge.clip_laion2b_soup_ft_in12k --data_name WikiArt --embedding_col_name convnext-clip --new_data_name convnext-clip

python3 src/classify.py --data_name convnext-clip --feature_col genre --embedding_col convnext-clip --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python3 src/classify.py --data_name convnext-clip --feature_col style --embedding_col convnext-clip --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python3 src/classify.py --data_name convnext-clip --feature_col artist --embedding_col convnext-clip --epochs 20 --hidden_layer_size 3000 --batch_size 32 

python3 src/classify.py --data_name eva02_clip_336 --feature_col genre --embedding_col eva02_clip_336 --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python3 src/classify.py --data_name eva02_clip_336 --feature_col style --embedding_col eva02_clip_336 --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python3 src/classify.py --data_name eva02_clip_336 --feature_col artist --embedding_col eva02_clip_336 --epochs 20 --hidden_layer_size 3000 --batch_size 32 

deactivate