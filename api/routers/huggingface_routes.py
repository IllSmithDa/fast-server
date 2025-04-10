from fastapi import APIRouter, File, UploadFile
from ..services.hugging_face import  generateDogFacts

router = APIRouter(
    prefix="/huggingface_routes",
    tags=["nerual network"],
    responses={404: {"description": "Not found"}},
)

@router.get("/get_dog_facts/{dog_breed}")
async def getDogFact(dog_breed: str):
    print(dog_breed)
    return await generateDogFacts(dog_breed)