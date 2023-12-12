source env/bin/activate # activate virtual environment

# run scripts
#python3 src/load_hf_dataset.py --huggingface_dataset huggan/wikiart --name WikiArt --seed 2830
#python3 src/extract_features.py --pretrained_model eva02_large_patch14_224.mim_m38m --data_name WikiArt --embedding_col_name eva02_large --new_data_name eva02_large
#python3 src/extract_features.py --pretrained_model convnextv2_huge.fcmae --data_name WikiArt --embedding_col_name convnextv2_huge --new_data_name convnextv2_huge
#python3 src/extract_features.py --pretrained_model convnext_base.clip_laiona --data_name WikiArt --embedding_col_name conv_clip_laiona --new_data_name conv_clip_laiona

# classify genre
#python3 src/classify.py --data_name eva02_large --feature_col genre --embedding_col eva02_large --epochs 20 --hidden_layer_size 3000 --batch_size 32 
#python3 src/classify.py --data_name conv_clip_laiona --feature_col genre --embedding_col conv_clip_laiona --epochs 20 --hidden_layer_size 3000 --batch_size 32 
#python3 src/classify.py --data_name convnextv2_huge --feature_col genre --embedding_col convnextv2_huge --epochs 20 --hidden_layer_size 3000 --batch_size 32 

# classify style
#python3 src/classify.py --data_name eva02_large --feature_col style --embedding_col eva02_large --epochs 20 --hidden_layer_size 3000 --batch_size 32 
#python3 src/classify.py --data_name conv_clip_laiona --feature_col style --embedding_col conv_clip_laiona --epochs 20 --hidden_layer_size 3000 --batch_size 32 
#python3 src/classify.py --data_name convnextv2_huge --feature_col style --embedding_col convnextv2_huge --epochs 20 --hidden_layer_size 3000 --batch_size 32 

# classify artist
python3 src/classify.py --data_name eva02_large --feature_col artist --embedding_col eva02_large --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python3 src/classify.py --data_name conv_clip_laiona --feature_col artist --embedding_col conv_clip_laiona --epochs 20 --hidden_layer_size 3000 --batch_size 32 
#python3 src/classify.py --data_name convnextv2_huge --feature_col artist --embedding_col convnextv2_huge --epochs 20 --hidden_layer_size 3000 --batch_size 32 

deactivate # deactivate virtual environment again