# load wikiart dataset from huggingface
python src/load_hf_dataset.py --huggingface_dataset huggan/wikiart --name WikiArt --seed 2830

# extract features
python src/extract_features.py --pretrained_model beit_large_patch16_512.in22k_ft_in22k_in1k --data_name WikiArt --embedding_col_name beit_large --new_data_name beit_large
python src/extract_features.py --pretrained_model eva02_large_patch14_224.mim_m38m --data_name WikiArt --embedding_col_name eva02_large --new_data_name eva02_large
python src/extract_features.py --pretrained_model eva02_large_patch14_clip_336.merged2b --data_name WikiArt --embedding_col_name eva02_clip_336 --new_data_name eva02_clip_336
python src/extract_features.py --pretrained_model convnextv2_huge.fcmae --data_name WikiArt --embedding_col_name convnextv2_huge --new_data_name convnextv2_huge
python src/extract_features.py --pretrained_model convnext_base.clip_laiona --data_name WikiArt --embedding_col_name conv_clip_laiona --new_data_name conv_clip_laiona

# classify genre
python src/classify.py --data_name beit_large --feature_col genre --embedding_col beit_large --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name eva02_clip_336 --feature_col genre --embedding_col eva02_clip_336 --epochs 20 --hidden_layer_size 3000 --batch_size 32
python src/classify.py --data_name eva02_large --feature_col genre --embedding_col eva02_large --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name conv_clip_laiona --feature_col genre --embedding_col conv_clip_laiona --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name convnextv2_huge --feature_col genre --embedding_col convnextv2_huge --epochs 20 --hidden_layer_size 3000 --batch_size 32 

# classify style
python src/classify.py --data_name beit_large --feature_col style --embedding_col beit_large --epochs 20 --hidden_layer_size 3000 --batch_size 32
python src/classify.py --data_name eva02_clip_336 --feature_col style --embedding_col eva02_clip_336 --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name eva02_large --feature_col style --embedding_col eva02_large --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name conv_clip_laiona --feature_col style --embedding_col conv_clip_laiona --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name convnextv2_huge --feature_col style --embedding_col convnextv2_huge --epochs 20 --hidden_layer_size 3000 --batch_size 32 

# classify artist
python src/classify.py --data_name beit_large --feature_col artist --embedding_col beit_large --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name eva02_clip_336 --feature_col artist --embedding_col eva02_clip_336 --epochs 20 --hidden_layer_size 3000 --batch_size 32
python src/classify.py --data_name eva02_large --feature_col artist --embedding_col eva02_large --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name conv_clip_laiona --feature_col artist --embedding_col conv_clip_laiona --epochs 20 --hidden_layer_size 3000 --batch_size 32 
python src/classify.py --data_name convnextv2_huge --feature_col artist --embedding_col convnextv2_huge --epochs 20 --hidden_layer_size 3000 --batch_size 32 
