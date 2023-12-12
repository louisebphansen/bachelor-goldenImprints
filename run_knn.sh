source env/bin/activate

python3 src/neighbors.py --data conv_clip_laiona_train --target_image 300 --feature_list conv_clip_laiona --plot_name conv_clip_laiona_knn_300.png

deactivate

