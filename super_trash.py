import datasets

eva224_test = datasets.load_from_disk("datasets/eva02_clip_224_test")
eva224_train = datasets.load_from_disk("datasets/eva02_clip_224_train")
eva224_val = datasets.load_from_disk("datasets/eva02_clip_224_val")

eva336_test = datasets.load_from_disk("datasets/eva02_clip_336_test")
eva336_train = datasets.load_from_disk("datasets/eva02_clip_336_train")
eva336_val = datasets.load_from_disk("datasets/eva02_clip_336_val")

## renaming 244
eva224_test = eva224_test.rename_column("eva02_large", "eva02_clip_224")
eva224_test.save_to_disk("datasets/eva02_clip_224_test_up")

eva224_train = eva224_train.rename_column("eva02_large", "eva02_clip_224")
eva224_train.save_to_disk("datasets/eva02_clip_224_train_up")

eva224_val = eva224_val.rename_column("eva02_large", "eva02_clip_224")
eva224_val.save_to_disk("datasets/eva02_clip_224_val_up")

## renaming 336
eva336_test = eva336_test.rename_column("eva02_large", "eva02_clip_336")
eva336_test.save_to_disk("datasets/eva02_clip_336_test_up")

eva336_train = eva336_train.rename_column("eva02_large", "eva02_clip_336")
eva336_train.save_to_disk("datasets/eva02_clip_336_train_up")

eva336_val = eva336_val.rename_column("eva02_large", "eva02_clip_336")
eva336_val.save_to_disk("datasets/eva02_clip_336_val_up")