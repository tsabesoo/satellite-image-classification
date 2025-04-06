import torch
from torch import nn
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import pickle

# Load the model (PyTorch version)
model_save_path = r"C:\\Users\\thath\\OneDrive - University of Bradford\\Discipline-specific Artificial Intelligence Project\\Project file\\Test 02\\model.pth"
model = torch.load(model_save_path)
model.eval()  # Set model to evaluation mode

# Load the label_to_index dictionary
label_to_index_save_path = r"C:\Users\thath\OneDrive - University of Bradford\Discipline-specific Artificial Intelligence Project\\Project file\\Test 02\\Testing code\\label_to_index.pkl"
with open(label_to_index_save_path, 'rb') as f:
    label_to_index = pickle.load(f)

# Create a reverse mapping from index to label
index_to_label = {index: label for label, index in label_to_index.items()}

# Define colors for each category
category_colors = {
    "AnnualCrop": (255, 255, 0, 128),  # Yellow with transparency
    "Forest": (0, 128, 0, 128),  # Green with transparency
    "HerbaceousVegetation": (144, 238, 144, 128),  # Light Green with transparency
    "Industrial": (255, 165, 0, 128),  # Orange with transparency
    "Pasture": (173, 216, 230, 128),  # Light Blue with transparency
    "PermanentCrop": (0, 100, 0, 128),  # Dark Green with transparency
    "Residential": (255, 0, 0, 128),  # Red with transparency
    "River": (0, 0, 255, 128),  # Blue with transparency
    "SeaLake": (0, 255, 255, 128)  # Cyan with transparency
}

# Preprocess the input image for PyTorch
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Use ImageNet mean and std
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Process the larger image and highlight the smaller images
def process_large_image(image_path):
    original_image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
    width, height = original_image.size
    final_image = original_image.copy()
    draw = ImageDraw.Draw(final_image, 'RGBA')

    for i in range(0, width, 64):
        for j in range(0, height, 64):
            box = (i, j, i + 64, j + 64)
            cropped_image = original_image.crop(box)
            preprocessed_image = preprocess_image(cropped_image)
            with torch.no_grad():  # No need to track gradients during inference
                outputs = model(preprocessed_image)  # Forward pass
                _, predicted_class = torch.max(outputs, 1)  # Get the predicted class index
                predicted_label = index_to_label[predicted_class.item()]

            # Get the color based on the predicted category
            color = category_colors.get(predicted_label, (255, 255, 255, 128))  # Default to semi-transparent white if category not found
            print(f"Box {box}: {predicted_label}, Color: {color}")

            # Highlight the image according to its category
            draw.rectangle(box, fill=color)

    return final_image

# Display the final image
image_path = r"C:\\Users\\thath\\OneDrive - University of Bradford\\Discipline-specific Artificial Intelligence Project\\Test sampel\\Screenshot 2025-03-14 160213.jpg"
final_image = process_large_image(image_path)
final_image.show()

# For testing with another image
image_path_2 = r"C:\\Users\\thath\\OneDrive - University of Bradford\\Discipline-specific Artificial Intelligence Project\\Test sampel\\test_large.jpg"
final_image_2 = process_large_image(image_path_2)
final_image_2.show()
