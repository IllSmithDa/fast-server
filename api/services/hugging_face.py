from huggingface_hub import InferenceClient, login
from transformers import pipeline
import os
import torch
  
key = os.getenv("HUGGINGFACE_TOKEN")
login(token=key)


client = InferenceClient(
    api_key=key,
)


async def generateDogFacts(dog_breed: str):
  prompt = f"Simply give me a short paragrah about the dog breed {dog_breed} without any introductions or conclusions."

  response = client.text_generation(prompt=prompt, model="mistralai/Mistral-7B-Instruct-v0.1")

  return {"dog_facts": response}