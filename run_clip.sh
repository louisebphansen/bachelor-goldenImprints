# extract features
# python3 src/extract_features.py --pretrained_model eva02_large_patch14_clip_224.merged2b --data_name WikiArt --embedding_col_name eva02_clip_224 --new_data_name eva02_clip_224
# python3 src/extract_features.py --pretrained_model eva02_large_patch14_clip_336.merged2b --data_name WikiArt --embedding_col_name eva02_clip_336 --new_data_name eva02_clip_336

# classify genre
python src/classify.py --data_name eva02_clip_224 --feature_col genre --embedding_col eva02_clip_224 --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name eva02_clip_336 --feature_col genre --embedding_col eva02_clip_336 --epochs 20 --hidden_layer_size 3000 --batch_size 32 

# classify style
python src/classify.py --data_name eva02_clip_224 --feature_col style --embedding_col eva02_clip_224 --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name eva02_clip_336 --feature_col style --embedding_col eva02_clip_336 --epochs 20 --hidden_layer_size 3000 --batch_size 32 

# classify artist
python src/classify.py --data_name eva02_clip_224 --feature_col artist --embedding_col eva02_clip_224 --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name eva02_clip_336 --feature_col artist --embedding_col eva02_clip_336 --epochs 20 --hidden_layer_size 3000 --batch_size 32
