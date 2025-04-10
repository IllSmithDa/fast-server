from fastapi import APIRouter, File, UploadFile
from ..services.effnet_b2 import predict

router = APIRouter(
    prefix="/torchvision_routes",
    tags=["nerual network"],
    responses={404: {"description": "Not found"}},
)

@router.post("/predict")
async def make_prediction(file: UploadFile = File(...)):
    return await predict(file)