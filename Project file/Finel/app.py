from pathlib import Path
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import time
import base64


from fastapi.responses import JSONResponse
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from tensorflow.keras.models import load_model
import pickle

# -----------------------------------------------------------------------------
# Configuration – everything is relative to this file so you can keep the whole
# project self‑contained. Place `model.keras` and `label_to_index.pkl` next to
# this script.
# -----------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent
MODEL_PATH    = BASE_DIR / "model.keras"
LABEL_MAP_PATH = BASE_DIR / "label_to_index.pkl"
OUTPUT_DIR    = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"[INIT] Base directory     : {BASE_DIR}")
print(f"[INIT] Model path         : {MODEL_PATH}")
print(f"[INIT] Label map path     : {LABEL_MAP_PATH}")
print(f"[INIT] Output directory   : {OUTPUT_DIR}")

# -----------------------------------------------------------------------------
# One‑time initialisation (executed when the server starts)
# -----------------------------------------------------------------------------
try:
    t0 = time.time()
    model = load_model(MODEL_PATH)
    print(f"[INIT] ✅ Model loaded in {time.time() - t0:.2f}s")
except Exception as e:
    raise RuntimeError(f"[INIT] ❌ Could not load model: {e}")

try:
    with open(LABEL_MAP_PATH, "rb") as f:
        label_to_index = pickle.load(f)
    print(f"[INIT] ✅ Label map loaded – {len(label_to_index)} classes")
except Exception as e:
    raise RuntimeError(f"[INIT] ❌ Could not load label map: {e}")

index_to_label = {v: k for k, v in label_to_index.items()}

CATEGORY_COLORS = {
    "AnnualCrop": (255, 255, 0, 128),
    "Forest": (0, 128, 0, 128),
    "HerbaceousVegetation": (144, 238, 144, 128),
    "Industrial": (255, 165, 0, 128),
    "Pasture": (173, 216, 230, 128),
    "PermanentCrop": (0, 100, 0, 128),
    "Residential": (255, 0, 0, 128),
    "River": (0, 0, 255, 128),
    "SeaLake": (0, 255, 255, 128),
}

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

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize to 64×64, normalise to 0‑1 and add batch dimension."""
    img = img.resize((64, 64))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def process_large_image(pil_image: Image.Image) -> Image.Image:
    """Slide a 64×64 window across the image, classify each tile, and overlay
    a semi‑transparent colour depending on the predicted class."""
    original = pil_image.convert("RGB")
    w, h = original.size
    print(f"[PROC] Image size: {w}×{h}")
    result = original.copy()
    draw = ImageDraw.Draw(result, "RGBA")

    tiles_x = (w + 63) // 64
    tiles_y = (h + 63) // 64
    total_tiles = tiles_x * tiles_y
    processed_tiles = 0
    t0 = time.time()

    for x in range(0, w, 64):
        for y in range(0, h, 64):
            box = (x, y, x + 64, y + 64)
            tile = original.crop(box)
            preds = model.predict(preprocess_image(tile), verbose=0)
            cls_idx = int(np.argmax(preds, axis=1)[0])
            label = index_to_label.get(cls_idx, "Unknown")
            colour = CATEGORY_COLORS.get(label, (255, 255, 255, 128))
            draw.rectangle(box, fill=colour)
            processed_tiles += 1
            if processed_tiles % 100 == 0 or processed_tiles == total_tiles:
                print(f"[PROC] Processed {processed_tiles}/{total_tiles} tiles")

    print(f"[PROC] Finished in {time.time() - t0:.2f}s")
    return result

# Calculate and display category changes
def calculate_category_changes(counts1, counts2):
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())

    changes = {}
    for category in counts1:
        percentage1 = (counts1[category] / total1) * 100 if total1 > 0 else 0
        percentage2 = (counts2[category] / total2) * 100 if total2 > 0 else 0
        changes[category] = (percentage1, percentage2, percentage2 - percentage1)

    return changes


def process_large_image_(image):
    original_image = image.convert('RGB')  # Ensure image is in RGB mode
    width, height = original_image.size
    final_image = original_image.copy()
    draw = ImageDraw.Draw(final_image, 'RGBA')

    category_counts = {label: 0 for label in index_to_label.values()}

    for i in range(0, width, 64):
        for j in range(0, height, 64):
            box = (i, j, i + 64, j + 64)
            cropped_image = original_image.crop(box)
            preprocessed_image = preprocess_image(cropped_image)
            predictions = model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = index_to_label[predicted_class]

            # Count the occurrences of each category
            category_counts[predicted_label] += 1

            # Debugging: Print the predicted label and corresponding color
            color = category_colors.get(predicted_label, (255, 255, 255, 128))  # Default to semi-transparent white if category not found
            print(f"Box {box}: {predicted_label}, Color: {color}")

            # Highlight the image according to its category
            draw.rectangle(box, fill=color)

    return final_image, category_counts


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Land‑cover heat‑map API", version="1.0")
print("[INIT] 🚀 FastAPI application created")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)



@app.get("/")
def root():
    return {"message": "API is up. POST an image to /overlay to get coloured overlay."}


@app.post("/overlay", response_class=StreamingResponse)
async def overlay_image(file: UploadFile = File(...)):
    """Accepts an image file (JPEG/PNG). Returns a PNG with class‑colour overlay and saves it."""
    print(f"[REQ ] Received file: {file.filename} (type: {file.content_type})")
    if file.content_type not in {"image/jpeg", "image/png"}:
        print("[REQ ] Unsupported file type")
        raise HTTPException(status_code=415, detail="Unsupported file type. Use JPEG or PNG.")

    contents = await file.read()
    try:
        img = Image.open(BytesIO(contents))
        print("[REQ ] Image opened successfully")
    except Exception as e:
        print(f"[REQ ] Cannot open image: {e}")
        raise HTTPException(status_code=400, detail="Cannot open image.")

    processed = process_large_image(img)

    # ------------------- Save processed image -------------------
    output_name = f"{Path(file.filename).stem}_overlay.png"
    output_path = OUTPUT_DIR / output_name
    processed.save(output_path, format="PNG")
    print(f"[SAVE] Processed image saved to: {output_path}")
    # ------------------------------------------------------------

    buf = BytesIO()
    processed.save(buf, format="PNG")
    buf.seek(0)
    print("[RESP] Returning processed image to client\n")
    return StreamingResponse(buf, media_type="image/png")


@app.post("/compare_images")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Accepts two image files, processes them, and returns a dictionary with combined image and category changes."""
    print(f"Received files: {file1.filename}, {file2.filename}")

    # Check file types
    if file1.content_type not in {"image/jpeg", "image/png"} or file2.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Unsupported file type. Use JPEG or PNG.")
    
    # Read image files
    contents1 = await file1.read()
    contents2 = await file2.read()
    
    try:
        img1 = Image.open(BytesIO(contents1))
        img2 = Image.open(BytesIO(contents2))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Cannot open image.")
    
    # Process images and calculate category changes
    predicted_image1, counts1 = process_large_image_(img1)
    predicted_image2, counts2 = process_large_image_(img2)
    
    changes = calculate_category_changes(counts1, counts2)
    
    # Combine images
    width1, height1 = predicted_image1.size
    width2, height2 = predicted_image2.size
    combined_width = width1 + width2
    combined_height = max(height1, height2)
    combined_image = Image.new('RGB', (combined_width, combined_height))
    combined_image.paste(predicted_image1, (0, 0))
    combined_image.paste(predicted_image2, (width1, 0))

    # Convert combined image to bytes for response
    buf = BytesIO()
    combined_image.save(buf, format="PNG")
    buf.seek(0)

    # Prepare response
    response = {
        "combined_image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "category_changes": changes
    }

        # Calculate and display category changes
    changes = calculate_category_changes(counts1, counts2)
    for category, (percentage1, percentage2, change) in changes.items():
        print(f"{category}: {percentage1:.2f}% -> {percentage2:.2f}% (Change: {change:.2f}%)")


    return JSONResponse(content=response)


# -----------------------------------------------------------------------------
# Optional: run with `python app.py` for quick testing (or use uvicorn directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("[MAIN] Starting development server on http://0.0.0.0:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
