import os

# "test" for Validation data
# "train" for training data
# "val" for validation data
path = r"D:\imagenet-object-localization-challenge\val"

for folders, subfolders, images in os.walk(path):
    for img in images:
        with open(r"D:\Study\My files\projects\image colorization\val_data.txt", "a") as f:
            f.write(img + "," + os.path.join(folders, img) + "\n")
        
                