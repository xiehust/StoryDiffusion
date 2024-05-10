from fastapi import FastAPI, Request
import uvicorn, json, datetime
import torch
from pydantic import BaseModel,Field
from typing import Literal,List, Any
from io import BytesIO
import base64
from PIL import Image
from diffusers import DiffusionPipeline

pipe = None
DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def load_model():
    global pipe
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")

def base64_to_image(base64_string):
    """
    Converts a base64 string to a PIL.Image.Image object.
    
    Args:
        base64_string (str): The base64 string representation of the image.
        
    Returns:
        PIL.Image.Image: The image object.
    """
    image_bytes = base64.b64decode(base64_string)
    image_buffer = BytesIO(image_bytes)
    image = Image.open(image_buffer)
    return image


def image_to_base64(image):
    """
    Converts a PIL.Image.Image object to a base64 string.
    
    Args:
        image (PIL.Image.Image): The image object to be converted.
        
    Returns:
        str: The base64 string representation of the image.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # Use format="JPEG" for JPEG images
    image_bytes = buffer.getvalue()
    base64_string = base64.b64encode(image_bytes).decode("utf-8")
    return base64_string


class APIRequest(BaseModel):
    num_steps : int = Field(default=50)
    cfg_scale: float = Field(default=7)
    negative_prompt : str =  Field(default="naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation")
    prompt: str
    G_height: int = Field(default=1024)
    G_width: int = Field(default=1024)
    style_preset: str
    
class APIResponse(BaseModel):
    images_base64 : List[str] = Field(..., descrition="Generated images")
    
app = FastAPI()

@app.get("/ping")
def ping():
    return {'status': 'Healthy'}

@app.post("/invocations")
async def create_item(request: APIRequest):
    global pipe
    if pipe is None:
        load_model()
    print(request.prompt)
    prompt=request.prompt + ' ' + request.style_preset 
    negative_prompt = request.negative_prompt
    width = request.G_width
    height = request.G_height
    images = pipe(prompt=prompt,
                  negative_prompt=negative_prompt,
                  width=width,
                  height=height,
                  target_size=(height,width),
                  original_size=(1024,1024)).images
    
    data = [image_to_base64(img) for img in images]

    return APIResponse(images_base64=data)


if __name__ == '__main__':
    load_model()
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
    print('server start')