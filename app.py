import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch import nn
from models import ResNet50, TinyVGG
from PIL import Image
import torchvision.transforms as transforms
import os

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] # Fix for torch classes path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class_names = ['Pizza', 'Steak', 'Sushi']
models = ['TinyVGG','ResNet50']


st.title("Image Classification with mutiple models")

# model_name = st.selectbox("Select a model", models, index=1)
# if model_name == 'TinyVGG':
#     model = torch.load("trained_models/model_TinyVGG_pizza_steak_sushi_v1.pth", weights_only=False, map_location=torch.device(device))
# elif model_name == 'ResNet50':
#     model = torch.load("trained_models/model_ResNet50_pizza_steak_sushi_v1.pth", weights_only=False, map_location=torch.device(device))

st.write(f"Upload an image to classify it into one of the following categories: {class_names}.")

img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img is not None:
    # Read the image and convert it to a tensor
    image = Image.open(img).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        col1.header("TinyVGG")
        model = torch.load("trained_models/model_TinyVGG_pizza_steak_sushi_v1.pth", weights_only=False, map_location=torch.device(device))
        # Make predictions
        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]

        st.write(f"Predicted class: **{predicted_class}**")
        st.write(f"Model confidence: **{torch.nn.functional.softmax(output, dim=1).max().item() * 100:.2f}%**")

    with col2:
        col2.header("ResNet50")
        model = torch.load("trained_models/model_ResNet50_pizza_steak_sushi_v1.pth", weights_only=False, map_location=torch.device(device))
        # Make predictions
        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]

        st.write(f"Predicted class: **{predicted_class}**")
        st.write(f"Model confidence: **{torch.nn.functional.softmax(output, dim=1).max().item() * 100:.2f}%**") 