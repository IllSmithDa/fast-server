import torchvision
import torch

# set device to gpu
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(gpu_device)


weights = torchvision.models.ResNet50_Weights.DEFAULT # .DEFAULT = best available weights 
model_0 = torchvision.models.resnet50(weights=weights).to(gpu_device)
