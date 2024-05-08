from fastapi import FastAPI, Request
import uvicorn, json, datetime
import torch
from gradio_app_sdxl_specific_id_low_vram import process_generation
from pydantic import BaseModel,Field
from typing import Literal,List, Any
import logging
logger = logging.getLogger(__name__)

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class APIRequest(BaseModel):
    sd_type: Literal['RealVision','SDXL','Unstable'] = Field(default='SDXL')
    modeltype : Literal["Only Using Textual Description","Using Ref Images"] = Field(default="Only Using Textual Description")
    files: Any = Field(default=None)
    num_steps : int = Field(default=50)
    style : Literal["Japanese Anime","(No style)","Cinematic","Disney Charactor","Photographic","Comic book","Line art"] = Field(default="Comic book")
    Ip_Adapter_Strength : float = Field(default=0.5, descrition="The strength of the IP adapter. The value ranges from 0 to 1. The larger the value, the stronger the IP adapter.")
    style_strength_ratio : int = Field(default=20 ,descrition="Style strength of Ref Image (%)")
    guidance_scale: float = Field(default=5.0)
    seed_: int = Field(default=0)
    sa32_:float = Field(default=0.5)
    sa64_:float = Field(default=0.5)
    id_length_ : int = Field(default=2,descrition="Number of id images in total images")
    general_prompt : str = Field(...,descrition="Textual Description for Character")
    negative_prompt : str =  Field(default="naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation")
    prompt_array: str =  Field(...,descrition="Comic Description (each line corresponds to a frame)")
    G_height: int = Field(default=512)
    G_width: int = Field(default=512)
    comic_type : Literal['No typesetting (default)','Four Pannel','Classic Comic Style'] = Field(default='Classic Comic Style')
    font_choice :str = Field(default='Inkfree.ttf')
    
app = FastAPI()

@app.get("/ping")
def ping():
    return {'status': 'Healthy'}

@app.post("/invocations")
async def create_item(request: APIRequest):
    json_post = request.json()
    json_post_raw = json.dumps(json_post)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    log = f"[ {time} ] - [Request]:{json_post_raw}"
    print(log)
    
    images = process_generation(request.sd_type,request.modeltype,request.files,request.num_steps,request.style,request.Ip_Adapter_Strength,request.style_strength_ratio,request.guidance_scale,request.seed_,request.sa32_,request.sa64_,request.id_length_,request.general_prompt,request.negative_prompt,request.prompt_array,request.G_height,request.G_width,request.comic_type,request.font_choice)
    print('--------process_generation done')
    print(images)
    torch_gc()
    return images


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
    logger.info('server start')