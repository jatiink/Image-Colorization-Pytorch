import torch
from torchvision import transforms
import cv2
import dataset   
import numpy as np 
import matplotlib.pyplot as plt

image_path = r"C:\Users\jatin\Downloads\istockphoto-952103152-612x612.jpg"
image = cv2.imread(image_path,0)
h, w = image.shape[0], image.shape[1]
in_img = dataset.transform(image)
in_img = in_img.reshape((1, 1, 256, 256))

def pred_img_visual(input_img, pred_img):
    in_img = dataset.inv_normalize(input_img)
    t = transforms.Resize((h,w))

    in_img = t(in_img)
    pred_img = t(pred_img)

    in_img = in_img.reshape((1, h, w))
    pred_img = pred_img.reshape((2, h, w))

    in_img = dataset.to_image(in_img)
    pred_img = dataset.to_image(pred_img)

    pred_img = np.dstack((in_img, pred_img))
    in_img = np.dstack((in_img, in_img, in_img))

    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_LAB2RGB)

    img = np.concatenate([in_img, pred_img], axis=1)
    plt.imshow(img)
    plt.show()

model = model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=1, out_channels=2, init_features=32, pretrained=False)
model.load_state_dict(torch.load('model_saves\Epoch_5.pt', map_location=torch.device('cpu')))
model.eval()

def prediction():
    with torch.no_grad():
        pred = model(in_img)

    pred_img_visual(in_img, pred)

if __name__ == '__main__':
    prediction()
