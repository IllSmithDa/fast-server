


from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

# have to declare another env path for the api folder
env_path = Path("api/.env")
# Load the .env file
load_dotenv(dotenv_path=env_path)

hive_ai_api_key = os.getenv("HIVE_AI_API_KEY")

# Configure the client with custom base URL and API key
client = OpenAI(
    base_url="https://api.thehive.ai/api/v3/",  # Hive AI's endpoint
    api_key=hive_ai_api_key  # Replace with your API key
)

async def get_completion(prompt, model="meta-llama/llama-3.2-1b-instruct"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    # Extract the response content
    return response.choices[0].message.content

async def generate_facts(dog_breed: str):
    result = await get_completion(f"I need a short paragraph about the dog breed {dog_breed} without any introductions or conclusions.")
    return {"dog_facts": result}