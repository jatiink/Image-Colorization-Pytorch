import cv2
import torch
import streamlit as st
import matplotlib.pyplot as plt
from dataset import *   
from visual import * 

# Load the model once when the app starts
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=1, out_channels=2, init_features=32, pretrained=False)
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()

def prediction(in_img, h, w):
    with torch.no_grad():
        pred = model(in_img)
    return pred_img_visual(in_img, pred, h, w)

# Streamlit app layout
st.title("Image Colorization")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    if st.button("Colorize"):
        with st.spinner("Colorizing..."):         
            # Read and process the image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 0)
            h, w = image.shape[0], image.shape[1]
            in_img = transform(image)
            in_img = in_img.reshape((1, 1, 256, 256))

            # Get predictions
            images = prediction(in_img, h, w)
            st.image(images["pred_image"], caption="Predicted Image", use_column_width=True)

            # Show download button after displaying the image
            im1 = cv2.cvtColor(images["pred_image"], cv2.COLOR_RGB2BGR)
            # Use BytesIO to save the image in memory
            _, buffer = cv2.imencode('.jpg', im1)
            byte_image = buffer.tobytes()
            # Save the image with a new name
            st.download_button("Download Image", byte_image, f"{uploaded_file.name.split('.')[0]}_colored.jpg", "image/jpeg")

