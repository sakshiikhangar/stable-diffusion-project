import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.unet.load_attn_procs("output_lora")

prompt = "MRI scan of brain with glioma tumor, btumr_style."

img = pipe(prompt, num_inference_steps=20, guidance_scale=6.5).images[0]
img.save("optimized_output.png")

print("Saved optimized_output.png")
