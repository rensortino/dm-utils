from diffusers import StableDiffusionPipeline
import torch
from torchvision.transforms.functional import to_tensor
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--cls", type=int, default=0)
parser.add_argument("--concept", type=int, default="class01")
args = parser.parse_args()

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_textual_inversion(f"learned_concepts/{args.concept}")
vae = pipeline.vae
class_id = "".join([i for i in args.concept.split() if i.isdigit()])

final_latents = []
reconstructed_images = []

for i in range(1000):
    image = pipeline(f"A photo of a <{args.concept}>", num_inference_steps=50).images[0]
    image = to_tensor(image).unsqueeze(0).to("cuda").half()
    with torch.no_grad():
        latent = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
        final_latents.append((latent.cpu(), args.cls))
        reconstructed = vae.decode(latent / vae.config.scaling_factor)
        reconstructed = to_pil_image(reconstructed[0][0].cpu().clamp(0,1))
        reconstructed_images.append((rencostructed, args.cls))

torch.save(final_latents, f"out/latents_{class_id}.pt")
torch.save(reconstructed_images, f"out/images_{class_id}.pt")