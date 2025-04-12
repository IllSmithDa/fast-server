from fastapi import APIRouter, File, UploadFile
from ..services.hive_ai import  generate_facts

router = APIRouter(
    prefix="/hive_ai_routes",
    tags=["nerual network"],
    responses={404: {"description": "Not found"}},
)


@router.get("/get_dog_facts/{dog_breed}")
async def getDogFact(dog_breed: str):
    print(dog_breed)
    return await generate_facts(dog_breed)