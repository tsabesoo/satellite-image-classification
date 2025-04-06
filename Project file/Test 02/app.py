from fastapi.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import io

# Load the model
model_save_path = r"C:\\Users\\thath\\OneDrive - University of Bradford\\Discipline-specific Artificial Intelligence Project\\Test model\\model.keras"
model = load_model(model_save_path)

# Load the label_to_index dictionary
label_to_index_save_path = r"C:\\Users\\thath\\OneDrive - University of Bradford\\Discipline-specific Artificial Intelligence Project\\label_to_index.pkl"
with open(label_to_index_save_path, 'rb') as f:
    label_to_index = pickle.load(f)

# Create a reverse mapping from index to label
index_to_label = {index: label for label, index in label_to_index.items()}

# Define colors for each category
category_colors = {
    "AnnualCrop": (255, 255, 0, 128),  # Yellow with transparency
    "Forest": (0, 128, 0, 128),  # Green with transparency
    "HerbaceousVegetation": (144, 238, 144, 128),  # Light Green with transparency
    "Highway": (128, 128, 128, 128),  # Gray with transparency
    "Industrial": (255, 165, 0, 128),  # Orange with transparency
    "Pasture": (173, 216, 230, 128),  # Light Blue with transparency
    "PermanentCrop": (0, 100, 0, 128),  # Dark Green with transparency
    "Residential": (255, 0, 0, 128),  # Red with transparency
    "River": (0, 0, 255, 128),  # Blue with transparency
    "SeaLake": (0, 255, 255, 128)  # Cyan with transparency
}

# Preprocess the input image
def preprocess_image(image):
    img = image.resize((64, 64))  # Resize image to match the input size of the model
    img_array = np.array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Process the larger image and highlight the smaller images
def process_large_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    final_image = image.copy()
    draw = ImageDraw.Draw(final_image, 'RGBA')

    for i in range(0, width, 64):
        for j in range(0, height, 64):
            box = (i, j, i + 64, j + 64)
            cropped_image = image.crop(box)
            preprocessed_image = preprocess_image(cropped_image)
            predictions = model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = index_to_label[predicted_class]

            # Get the color based on the predicted category
            color = category_colors.get(predicted_label, (255, 255, 255, 128))  # Default to semi-transparent white if category not found
            print(f"Box {box}: {predicted_label}, Color: {color}")

            # Highlight the image according to its category
            draw.rectangle(box, fill=color)

    return final_image

# Initialize FastAPI app
app = FastAPI()

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Process the image
    final_image = process_large_image(image)

    # Save the processed image to a byte stream
    img_byte_arr = io.BytesIO()
    final_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)  # Go to the beginning of the byte stream

    # Return the image as a response
    return StreamingResponse(img_byte_arr, media_type="image/png")