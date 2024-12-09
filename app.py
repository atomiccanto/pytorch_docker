from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import torch
import torchvision.models as models
from PIL import Image
from io import BytesIO
from torchvision import transforms
import json

app = FastAPI()

# Load ImageNet class labels
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(LABELS_URL)
class_idx = response.json()

model = models.resnet18(pretrained=True)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

class ImageRequest(BaseModel):
    image_url: str

# Image preprocessing function
def preprocess_image(image_url: str):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),  
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    return input_batch

@app.post("/classify/")
async def classify_image(request: ImageRequest):
    try:
        image_url = request.image_url
        input_batch = preprocess_image(image_url)
        with torch.no_grad():
            output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        result = []
        for i in range(top5_prob.size(0)):
            class_id = top5_catid[i].item()
            class_name = class_idx[str(class_id)][1]
            result.append({
                "class_id": class_id,
                "class_name": class_name,
                "probability": top5_prob[i].item()
            })

        return {"top_5_predictions": result}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
