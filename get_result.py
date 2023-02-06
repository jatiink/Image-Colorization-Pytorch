import cv2
import torch
import argparse
import matplotlib.pyplot as plt
from dataset import *   
from visual import * 

parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default="image.jpg", help="Image Path")
parser.add_argument("--show", action="store_true", help="Show Concatenated Images")
parser.add_argument("--save-output", action="store_true", help="Saves Output Image")
parser.add_argument("--save-conc", action="store_true", help="Saves Concatenated Images")
opt = parser.parse_args()

image_path = opt.img

image = cv2.imread(image_path, 0)
h, w = image.shape[0], image.shape[1]
in_img = transform(image)
in_img = in_img.reshape((1, 1, 256, 256))

model = model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=1, out_channels=2, init_features=32, pretrained=False)
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()

def prediction():
    with torch.no_grad():
        pred = model(in_img)
    return pred_img_visual(in_img, pred, h, w)

if __name__ == '__main__':
    images = prediction()
    
    if opt.show:
        plt.imshow(images["conc_image"])
        plt.show()
        
    if opt.save_output:
        im1 = cv2.cvtColor(images["pred_image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite("Output_image.jpg", im1)
   
    if opt.save_conc:
        im2 = cv2.cvtColor(images["conc_image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite("Output_image.jpg", im2)