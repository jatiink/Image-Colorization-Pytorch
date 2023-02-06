from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import random
import cv2

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256,256)),
                                transforms.Normalize((0.5,), (0.5,))])

tar_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((256,256))])


class MyDataset(Dataset):

    def __init__(self, txt_path, transform = transform):

        txt = open(txt_path, 'r')

        data, names, paths = [], [], []

        for line in txt.readlines():
            txt = line.replace('\n', '')
            txt = txt.replace(' ', '')
            txt = txt.split(",")
            data.append(txt)

        for img_names, img_paths in data:
            names.append(img_names)
            paths.append(img_paths)

        self.paths = paths
        self.img_names = names
        self.transform = transform

    def __getitem__(self, indx):
        read_flag = False
        while not read_flag:
            try:
                image = io.imread(self.paths[indx])
                if image.shape[2] == 3:
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    read_flag = True
                elif image.shape[2] == 4:
                    img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                    read_flag = True
                else:
                    indx = random.randint(0, len(self.img_names)-1)
            except Exception as e:
                print(e)
                indx = random.randint(0, len(self.img_names)-1)

        input_img = lab[: , :, :1]
        target = lab[:, :, 1:]
        input_img = transform(input_img)
        target = tar_transform(target)

        name = self.img_names[indx]

        inputs = {"image" : input_img,
                  "target" : target,
                  "name": name}

        return inputs

    def __len__(self):
        return len(self.img_names)


