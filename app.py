import os
import torch
import base64
from io import BytesIO
from torch import autocast
import random
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    repo_id = "dreamlike-art/dreamlike-diffusion-1.0"
    scheduler = EulerDiscreteScheduler.from_pretrained(
        repo_id, 
        subfolder="scheduler", 
        prediction_type="epsilon"
    )
    model = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, scheduler=scheduler).to("cuda")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    random_number = random.randint(1000, 99000)
    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    negative = model_inputs.get('negative', None)
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    num_inference_steps = model_inputs.get('num_inference_steps', 20)
    guidance_scale = model_inputs.get('guidance_scale', 7)
    input_seed = model_inputs.get("seed", random_number)
    
    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    with autocast("cuda"):
        image = model(prompt, negative_prompt=negative, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
     
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'seed':random_number,'image_base64': image_base64}
