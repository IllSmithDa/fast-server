# model_server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from api.routers import torchvision_routes


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

# Find the .env file
env_path = find_dotenv()

# Load the .env file
load_dotenv(dotenv_path=env_path)

app.include_router(torchvision_routes.router)


@app.get("/")
def read_root():
    return {"message": "FastAPI is now up!"}
