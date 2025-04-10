# model_server.py
from fastapi import File, UploadFile
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import io
import torchvision
import torch
import requests
from io import BytesIO
from dotenv import load_dotenv
import os

# have to declare another env path for the api folder
env_path = Path("api/.env")
# Load the .env file
load_dotenv(dotenv_path=env_path)


# set device to gpu if possible or just do cpu
targeted_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get the labels
with open('api/dog_labels.txt', 'r') as f:
    labels = [line.strip() for line in f if line.strip()]


# setup model config
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT # .DEFAULT = best available weights 
model_0 = torchvision.models.efficientnet_b2(weights=weights).to(targeted_device)
num_features = model_0.classifier[1].in_features
model_0.classifier[1] = torch.nn.Linear(num_features, len(labels))


# gaet aws resource
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



async def predict(file: UploadFile = File(...)):
  img_bytes = await file.read()
  image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
  img_tensor = transform(image).unsqueeze(0)
  # make sure grad is off
  with torch.inference_mode():
      target_image_pred = model_0(img_tensor)
      target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
      predicted = torch.argmax(target_image_pred_probs, dim=1).item()
      return {"prediction": labels[predicted]}