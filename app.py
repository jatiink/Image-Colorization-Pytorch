import cv2
import torch
import streamlit as st
import matplotlib.pyplot as plt
from dataset import *   
from visual import * 

# Load model code remains same...
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=1, out_channels=2, init_features=32, pretrained=False)
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu'), weights_only=True))
model.eval()

def prediction(in_img, h, w):
    original_img = in_img
    in_img= transform(in_img)
    in_img = in_img.reshape((1, 1, 256, 256))
    with torch.no_grad():
        pred = model(in_img)
        print(pred.shape)
        image = np.concatenate((in_img, pred), axis=1)
        print(image.shape)
    return pred_img_visual(original_img, pred, h, w)

# Streamlit app layout
st.set_page_config()  # Use wide layout for better image display
st.title("Image Colorization")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    show_original = st.checkbox("Show original image")
    
    # Display uploaded image if checkbox is selected
    if show_original:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 0)
        st.image(image, caption="Original Image", use_container_width="auto")
        # Reset file pointer
        uploaded_file.seek(0)
        
    if st.button("Colorize"):
        with st.spinner("Colorizing..."):         
            # Read and process image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 0)
            h, w = image.shape[0], image.shape[1]
            images = prediction(image, h, w)
            
            # Display high-res image
            st.image(
                images["pred_image"],
                caption="Colored Image",
                use_container_width="auto",  # Maintains aspect ratio
                clamp=True,  # Ensures proper pixel value range
                output_format="PNG"  # Use PNG for better quality
            )


            # Show download button after displaying the image
            im1 = cv2.cvtColor(images["pred_image"], cv2.COLOR_RGB2BGR)
            # Use BytesIO to save the image in memory
            _, buffer = cv2.imencode('.jpg', im1)
            byte_image = buffer.tobytes()
            # Save the image with a new name
            st.download_button("Download Image", byte_image, f"{uploaded_file.name.split('.')[0]}_colored.jpg", "image/jpeg")
