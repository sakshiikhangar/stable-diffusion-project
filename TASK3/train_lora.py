import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model

# -----------------------------
# Dataset
# -----------------------------
class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder: str, resolution: int = 512):
        self.folder = Path(folder)
        self.images = sorted([p for p in self.folder.glob("*.jpg")] + [p for p in self.folder.glob("*.png")])
        self.captions = [p.with_suffix(".txt") for p in self.images]

        if len(self.images) == 0:
            raise ValueError(f"ERROR: Dataset folder is empty: {folder}")

        # images -> [0,1], then normalized to [-1,1]
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),  # float32 [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1, +1]
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = Image.open(self.images[idx]).convert("RGB")
        with open(self.captions[idx], "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return self.transform(img), caption


# -----------------------------
# Utilities
# -----------------------------
def print_model_debug(pipe):
    print("==== PIPELINE DEBUG ====")
    try:
        print("UNet in_channels:", pipe.unet.config.in_channels)
    except Exception as e:
        print("Could not read pipe.unet.config.in_channels:", e)
    try:
        print("VAE config in_channels (if available):", getattr(pipe.vae.config, "in_channels", "NA"))
    except Exception:
        pass
    print("========================")


# -----------------------------
# Training
# -----------------------------
def train(
    data_dir: str = r"sd_dataset",
    output_dir: str = "lora_brain_tumor",
    model_name: str = "runwayml/stable-diffusion-v1-5",
    resolution: int = 512,
    batch_size: int = 1,
    epochs: int = 3,
    lr: float = 5e-6,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    gradient_accumulation_steps: int = 1,
    device_str: str = None,
    max_grad_norm: float = 1.0,
):
    # device
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        # choose safest dtype for UNet: bf16 if supported, else fp16
        if torch.cuda.is_bf16_supported():
            unet_dtype = torch.bfloat16
        else:
            unet_dtype = torch.float16
    else:
        unet_dtype = torch.float32

    print("Device:", device, "UNet dtype:", unet_dtype)

    # Load pipeline (we'll move parts manually)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # load in fp32 and move parts explicitly below
        local_files_only=False,
        safety_checker=None,
    )

    # quick debug print
    print_model_debug(pipe)

    # Create LoRA config for UNet
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="UNET",
    )

    # Wrap UNet with PEFT LoRA (this returns a model with trainable adapter params)
    unet_lora = get_peft_model(pipe.unet, lora_config)

    # Move models to device with safe dtypes
    # UNet -> unet_dtype (fp16 or bf16 on CUDA)
    try:
        unet_lora.to(device)
        if unet_dtype in (torch.float16, torch.bfloat16):
            unet_lora.to(unet_dtype)
    except Exception as e:
        print("Warning: failed to cast UNet to desired dtype:", e)

    # Text encoder -> keep in fp32 for stability
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    text_encoder.to(device)
    text_encoder.to(torch.float32)

    # VAE -> keep in fp32 for stable encoding
    vae = pipe.vae
    vae.to(device)
    vae.to(torch.float32)

    # Ensure only LoRA params are trainable (PEFT should handle this)
    unet_lora.print_trainable_parameters()

    # Dataset and loader
    dataset = TextImageDataset(data_dir, resolution=resolution)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(unet_lora.parameters(), lr=lr)
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # common latent scaling factor used by SD pipelines
    vae_scale_factor = 0.18215

    # AMP scaler for fp16 training
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and unet_dtype == torch.float16))

    unet_lora.train()
    global_step = 0

    for epoch in range(epochs):
        for step, (images, captions) in enumerate(dataloader):
            # move images to device as fp32 (VAE expects fp32)
            images = images.to(device=device, dtype=torch.float32)

            # Tokenize captions
            tokenized = tokenizer(
                captions,
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(device)

            # Get encoder hidden states (keep text encoder in fp32)
            with torch.no_grad():
                encoder_outputs = text_encoder(input_ids)
            encoder_hidden_states = encoder_outputs.last_hidden_state.to(device=device, dtype=torch.float32)

            # Encode images -> latents (keep this in fp32 then cast to unet dtype)
            with torch.no_grad():
                latents_dist = vae.encode(images).latent_dist
                latents = latents_dist.sample().to(device=device, dtype=torch.float32)

            latents = latents * vae_scale_factor

            # cast latents & encoder states to unet dtype for UNet forward
            if unet_dtype != torch.float32:
                latents = latents.to(dtype=unet_dtype)
                encoder_hidden_states_for_unet = encoder_hidden_states.to(dtype=unet_dtype)
            else:
                encoder_hidden_states_for_unet = encoder_hidden_states

            # create noise in same dtype/device as latents
            noise = torch.randn_like(latents, device=device, dtype=latents.dtype)

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Debug print once
            if global_step == 0:
                print("---- DEBUG (first step) ----")
                print("images.shape:", images.shape, "images.dtype:", images.dtype)
                print("latents.shape:", latents.shape, "latents.dtype:", latents.dtype)
                print("noisy_latents.shape:", noisy_latents.shape, "noisy_latents.dtype:", noisy_latents.dtype)
                print("encoder_hidden_states.shape:", encoder_hidden_states_for_unet.shape, "dtype:", encoder_hidden_states_for_unet.dtype)
                print("unet param dtype example:", next(unet_lora.parameters()).dtype)
                print("-----------------------------")

            # Forward + loss with optional AMP
            optimizer.zero_grad()

            if device.type == "cuda" and unet_dtype == torch.float16:
                # use autocast for fp16 UNet
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    model_pred = unet_lora(noisy_latents, timesteps, encoder_hidden_states_for_unet).sample
                    loss = torch.nn.functional.mse_loss(model_pred, noise)
                # scale + backward
                scaler.scale(loss).backward()
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet_lora.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # fp32 or bf16 path
                model_pred = unet_lora(noisy_latents, timesteps, encoder_hidden_states_for_unet).sample
                loss = torch.nn.functional.mse_loss(model_pred, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet_lora.parameters(), max_grad_norm)
                optimizer.step()

            # NaN check
            if torch.isnan(loss).any():
                print(f"NaN detected at epoch {epoch} step {step}. Stopping training.")
                return

            if step % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {step}, Loss={loss.item():.6f}")

            global_step += 1

    # Save LoRA adapter in diffusers-friendly filename
    os.makedirs(output_dir, exist_ok=True)
    # PEFT save
    unet_lora.save_pretrained(output_dir)
    # PEFT often writes 'adapter_model.bin' — rename if needed for diffusers
    adapter_bin = Path(output_dir) / "adapter_model.bin"
    lora_bin = Path(output_dir) / "pytorch_lora_weights.bin"
    if adapter_bin.exists() and not lora_bin.exists():
        adapter_bin.rename(lora_bin)

    print(f"Training complete! LoRA saved to: {output_dir}")


if __name__ == "__main__":
    train()
