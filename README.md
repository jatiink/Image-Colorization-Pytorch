# Image-Colorization-Pytorch
#### In this project, I used the Imagenet Dataset and a U-net model. This model works with the `L` channel from LAB COLOR SPACE and can predict the `A` and `B` channels.<br />
I also used Tensorboard for live results of training and validation.

## INSTRUCTIONS
### This project requires the following libraries :

•	[io(Scikit-image)](https://scikit-image.org/)<br />
•	[Torch(Pytorch)](https://pytorch.org/docs/stable/index.html)<br />
•	[Numpy](https://numpy.org/)<br />
•	[Cv2(OpenCV)](https://docs.opencv.org/4.x/)<br />
•	[Matplotlib](https://matplotlib.org/stable/index.html)<br />
• [torch.utils.tensorboard](https://www.tensorflow.org/tensorboard/get_started)<br />

### Please ensure you have installed the following libraries mentioned above before continuing.<br />
#### To install the following libraries.
##### Activate your virtual environment and type:
`pip install -r requirement.txt`

## HOW TO Get results

To get results on new Image.<br />
Extract all files in one folder.<br />

#### Run CMD in folder directory and type:

##### To view only
```
python get_result.py --img "Image path"
```

##### To save output Image
```
python get_result.py --img "Image path" --save-output
```

##### To save concatenated Images
```
python get_result.py --img "Image path" --save-conc
```

##### Example:
```
python get_result.py --img_root "C:/user/data/img.jpg" --show --save-output
```
#### Output will save in same directory.
### Note: There is a limitations of colors in trained model.

## Training Results
![Untitled-1](https://user-images.githubusercontent.com/97089717/216937616-3efed870-183d-4a47-8d9d-cf70a793cb62.png)
