import torch, logging
## disable warnings
logging.disable(logging.WARNING)  
## Imaging  library
import json
## Basic libraries
from tqdm.auto import tqdm
import os
## For video display
## Import the CLIP artifacts 
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from utils import load_image, pil_to_latents, latents_to_pil, latents_to_torch_img, text_enc, text_emb
import argparse

model_id = "runwayml/stable-diffusion-v1-5"

# Load diffusion 
tokenizer = CLIPTokenizer.from_pretrained(model_id)
text_encoder = CLIPTextModel.from_pretrained(model_id).to("cuda:0")
token_embedding = text_encoder.text_model.embeddings
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae",).to("cuda:0").half()
scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda:0").half()

# args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--diff_steps', type=int, default=50)

args = parser.parse_args()
if not os.path.exists("out"):
    os.makedirs("out")

#hyperparams
with open("prompts/imagenet10.json") as f:
    prompts = json.load(f)

dim=512
g= 7.5
bs = 1

samples_per_class = 1000

# Converting textual prompts to embedding

# text = text_enc(prompts,tokenizer,text_encoder) 
# text_embedding = text_emb(prompts, tokenizer, token_embedding) 
final_latents = []

with torch.no_grad():
    for cls, prompt in prompts.items():
        
        prompt = [prompt] * bs
        text, inp_emb = text_enc(prompt, tokenizer, text_encoder) 

        # Adding an unconditional prompt , helps in the generation process
        uncond, _ =  text_enc([""] * bs,tokenizer,text_encoder, text.shape[1])
        emb = torch.cat([uncond, text]).half()

        for i in range(samples_per_class):
            if (i+1) % 100 == 0:
                torch.save(final_latents, f"out/latents_{int(cls)*(i+1) + (i+1)}.pt")
            # Initiating random noise
            latents = torch.randn((bs, unet.in_channels, dim//8, dim//8)).half()

            # Setting number of steps in scheduler
            scheduler.set_timesteps(args.diff_steps)

            # Adding noise to the latents 
            latents = latents.to("cuda:0") * scheduler.init_noise_sigma

            # Iterating through defined steps
            for j,ts in enumerate(tqdm(scheduler.timesteps)):

                inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)
                # Predicting noise residual using U-Net
                u,t = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
                # Performing Guidance
                pred = u + g*(t-u)
                # Conditioning  the latents
                latents = scheduler.step(pred, ts, latents).prev_sample
                # img = latents_to_pil(latents, vae)[0].save(f"tmp_{j}.png")

                # emb = text_encoder.text_model(inp.input_ids.to("cuda:0"), inp_emb)
            final_latents.append((latents.cpu(), int(cls)))

torch.save(final_latents, f"out/latents_final.pt")  