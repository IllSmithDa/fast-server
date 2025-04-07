# model_server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import torchvision
import torch
from dotenv import load_dotenv
import requests
from io import BytesIO
import os


# set device to gpu
targeted_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
model_0 = torchvision.models.efficientnet_b0(weights=weights).to(targeted_device)


app = FastAPI()

# CORS config
origins = [
    "http://localhost:5173",
    "http://localhost:5173/animal-client",
    "https://illsmithda.github.io/animal-client/",
    "https://illsmithda.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the .env file
load_dotenv()

model_url = os.getenv("MODEL_RESOURCE_URL")
response = requests.get(model_url)

response.raise_for_status()  # Optional: will throw error if download fails


buffer = BytesIO(response.content)

# Load model
model_0.load_state_dict(torch.load(buffer, map_location=torch.device(targeted_device)))
model_0.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])

with open('name of the animals.txt', 'r') as f:
    labels = [line.strip() for line in f if line.strip()]

@app.get("/")
def read_root():
    return {"message": "FastAPI is now up!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print('hello')
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    # make sure grad is off
    with torch.inference_mode():
        target_image_pred = model_0(img_tensor)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        predicted = torch.argmax(target_image_pred_probs, dim=1).item()

        return {"prediction": labels[predicted]}