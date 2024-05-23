import requests
from io import BytesIO
from PIL import Image
import numpy as np
import urllib.request
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys

import inspect

import jax
import numpy as np
import orbax.checkpoint
import tensorflow as tf
from absl import app, flags

import wandb
from susie.jax_utils import (
    initialize_compilation_cache,
)
from susie.model import create_sample_fn

# jax diffusion stuff
from absl import app as absl_app
from absl import flags
from PIL import Image
import jax
import jax.numpy as jnp

# flask app here
import base64
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

# Prismatic VLM
import requests
import torch
from PIL import Image
from pathlib import Path
from prismatic import load

# create rng
rng = jax.random.PRNGKey(0)

from datetime import datetime
import os
from collections import deque
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
from typing import Callable, List, Tuple
from flask import Flask, request, jsonify
import imageio
import jax
import numpy as np
from absl import app, flags
from openai import OpenAI
##############################################################################

# Diffusion model params 
CHECK_POINT_PATH = "gs://catg_central2/logs/susie-nav_2024.04.26_23.01.31/200000/state"
WANDB_NAME = "catglossop/susie/jxttu4lu"
PRETRAINED_PATH = "runwayml/stable-diffusion-v1-5:flax"
prompt_w = 5.0
context_w = 5.0
diffusion_num_steps = 50


# For gated LMs like Llama-2, make sure to request official access, and generate an access token
hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
model_id = "prism-dinosiglip+7b"
vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
user_prompt = "What is going on in this image?"

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate!
generated_text = vlm.generate(
    image,
    prompt_text,
    do_sample=True,
    temperature=0.4,
    max_new_tokens=512,
    min_length=1,
)


class GenerateSubgoals: 

    def __init__(self):


        # Make the diffusion model
        self.CHECK_POINT_PATH = "gs://catg_central2/logs/susie-nav_2024.04.26_23.01.31/200000/state"
        self.WANDB_NAME = "catglossop/susie/jxttu4lu"
        self.PRETRAINED_PATH = "runwayml/stable-diffusion-v1-5:flax"
        self.prompt_w = 5.0
        self.context_w = 5.0
        self.diffusion_num_steps = 50
        self.diffusion_sample = create_sample_fn(
                self.CHECK_POINT_PATH,
                self.WANDB_NAME,
                self.diffusion_num_steps,
                self.prompt_w,
                self.context_w,
                0.0,
                self.PRETRAINED_PATH,
            )
        
        # Load the dataset --> should be the annotated dataset 


        

        




