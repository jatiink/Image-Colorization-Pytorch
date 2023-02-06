import cv2
import numpy as np
from torchvision import transforms

inv_normalize = transforms.Compose([transforms.Normalize((0.,),(1/0.5,)),
                                    transforms.Normalize((-0.5,),(1.))])

def to_image(pic):
    npimg = pic.byte()
    npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))
    return (npimg*255).astype("uint8")

def get_visual(input, target, predictions):
    input = input.detach().cpu()
    predictions = predictions.detach().cpu()
    target = target.detach().cpu()
    img_list = []

    for i in range(input.shape[0]):
        img1 = input[i]
        img2 = target[i]
        img3 = predictions[i]

        img1 = inv_normalize(img1)

        img1 = to_image(img1)
        img2 = to_image(img2)
        img3 = to_image(img3)

        in_img = np.dstack((img1,img1,img1))
        tar_img = np.dstack((img1, img2))
        pred_img = np.dstack((img1, img3))

        tar_img = np.asarray(transforms.ToPILImage()(tar_img))
        pred_img = np.asarray(transforms.ToPILImage()(pred_img))

        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_LAB2RGB)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_LAB2RGB)

        img = np.concatenate([in_img, tar_img, pred_img], axis=1)
        img_list.append(img)

    return img_list


def pred_img_visual(input_img, pred_img, h, w):
    in_img = inv_normalize(input_img)
    t = transforms.Resize((h,w))

    in_img = t(in_img)
    pred_img = t(pred_img)

    in_img = in_img.reshape((1, h, w))
    pred_img = pred_img.reshape((2, h, w))

    in_img = to_image(in_img)
    pred_img = to_image(pred_img)

    pred_img = np.dstack((in_img, pred_img))
    in_img = np.dstack((in_img, in_img, in_img))

    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_LAB2RGB)

    img = np.concatenate([in_img, pred_img], axis=1)
    return {"pred_image": pred_img,
            "conc_image": img}