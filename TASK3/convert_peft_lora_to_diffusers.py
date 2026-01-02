import torch
import safetensors.torch as s
from safetensors.torch import save_file

input_path = "lora_brain_tumor/adapter_model.safetensors"
output_path = "lora_brain_tumor/diffusers_lora.safetensors"

state = s.load_file(input_path)

new_state = {}

for key, value in state.items():
    # Remove the PEFT prefix
    new_key = key.replace("base_model.model.", "")

    # Convert PEFT format → Diffusers format
    new_key = new_key.replace("attentions.", "attentions_")
    new_key = new_key.replace("transformer_blocks.", "transformer_blocks_")
    new_key = new_key.replace("attn1.", "attn1_")
    new_key = new_key.replace("attn2.", "attn2_")
    new_key = new_key.replace(".lora_A.weight", ".lora_A")
    new_key = new_key.replace(".lora_B.weight", ".lora_B")

    # Add the prefix expected by Diffusers
    new_key = "unet." + new_key

    new_state[new_key] = value

save_file(new_state, output_path)
print("Conversion complete! Saved:", output_path)
