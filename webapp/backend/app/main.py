# webapp/backend/app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, shutil, random
from PIL import Image, ImageDraw

app = FastAPI()

# ✅ CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

# ✅ Create folders if missing
folders = [
    "original", "segnet", "unetpp",
    "heatmap/segnet", "heatmap/unetpp"
]
for f in folders:
    os.makedirs(os.path.join(RESULT_DIR, f), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Serve static files for results
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

def sanitize_filename(filename: str):
    return filename.replace(" ", "_").replace("/", "_")

def simulate_segmentation(input_path, output_path, model_type):
    """Draw dummy segmentation marks"""
    im = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size

    for _ in range(5):
        x0 = random.randint(0, w//2)
        y0 = random.randint(0, h//2)
        x1 = x0 + random.randint(20, 50)
        y1 = y0 + random.randint(20, 50)
        color = "red" if model_type == "segnet" else "green"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

    im.save(output_path)

@app.get("/")
def root():
    return {"message": "Backend is running. Use /upload/ to POST images."}

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    filename = sanitize_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)

    # ✅ Save uploaded file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ✅ Save original
    shutil.copy(file_path, os.path.join(RESULT_DIR, "original", filename))

    # ✅ Simulate SegNet
    segnet_path = os.path.join(RESULT_DIR, "segnet", filename)
    simulate_segmentation(file_path, segnet_path, "segnet")

    # ✅ Simulate U-Net++
    unetpp_path = os.path.join(RESULT_DIR, "unetpp", filename)
    simulate_segmentation(file_path, unetpp_path, "unetpp")

    # ✅ Simulate heatmaps (for demo just copy original)
    shutil.copy(file_path, os.path.join(RESULT_DIR, "heatmap/segnet", filename))
    shutil.copy(file_path, os.path.join(RESULT_DIR, "heatmap/unetpp", filename))

    # ✅ Random metrics
    metrics = {
        "segnet": {
            "dice": round(random.uniform(0.7, 0.85), 2),
            "iou": round(random.uniform(0.6, 0.78), 2),
            "sensitivity": round(random.uniform(0.7, 0.9), 2),
            "specificity": round(random.uniform(0.6, 0.88), 2)
        },
        "unetpp": {
            "dice": round(random.uniform(0.8, 0.92), 2),
            "iou": round(random.uniform(0.65, 0.85), 2),
            "sensitivity": round(random.uniform(0.75, 0.92), 2),
            "specificity": round(random.uniform(0.65, 0.89), 2)
        }
    }

    return JSONResponse({"filename": filename, "metrics": metrics})
