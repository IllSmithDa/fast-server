# FAST API SERVER FOR A LLN

   1. Start up a server 

     - uvicorn --reload api.main:app --reload

     - https://stackoverflow.com/questions/76939674/fastapi-attempted-relative-import-beyond-top-level-package

   2. Update requirement.txt

     - pip freeze > requirements.txt

   3. Hugging Face docs for different providers

      - https://huggingface.co/docs/inference-providers/en/tasks/text-generation

   4. Using huggerface 

      - Huggerface route will not be used as it is currently out of credits

      - instead, a free alternative will be used to ensure the low cost of this simple application. 
      
# Purpose

  1. The purpose of this server is to consume the create LLM to predict real world images of dogs and classify its breed

  2. Fast Api quick setup made this ideal choice for this type of project