import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as T
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model


# ----------------------------------------------------------
# Dataset (FIXED)
# ----------------------------------------------------------
class ImageCaptionDataset(Dataset):
    def __init__(self, folder):
        self.images = []
        self.captions = []

        for f in os.listdir(folder):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder, f)
                txt_path = img_path.rsplit(".", 1)[0] + ".txt"
                if os.path.exists(txt_path):
                    self.images.append(img_path)
                    self.captions.append(txt_path)

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        img = self.transform(img)

        with open(self.captions[i], "r") as f:
            caption = f.read().strip()

        return img, caption


# ----------------------------------------------------------
# TRAINING
# ----------------------------------------------------------
def main():

    MODEL = "runwayml/stable-diffusion-v1-5"
    DATA_DIR = "sd_dataset"
    OUTPUT_DIR = "output_lora"

    device = "cuda"

    # Load models
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(device)

    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet")
    unet.enable_gradient_checkpointing()

    # LoRA config
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["to_q", "to_k", "to_v"]
    )

    unet = get_peft_model(unet, lora_config).to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    dataset = ImageCaptionDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)  # Windows fix

    noise_scheduler = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")
    vae_scale = 0.18215
    EPOCHS = 15

    scaler = torch.cuda.amp.GradScaler()

    unet.train()
    text_encoder.eval()

    for epoch in range(EPOCHS):
        for step, (images, captions) in enumerate(loader):
            images = images.to(device, dtype=torch.float32)

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * vae_scale

            tokens = tokenizer(
                captions, truncation=True, padding="max_length",
                max_length=77, return_tensors="pt"
            ).input_ids.to(device)

            with torch.no_grad():
                enc_hidden = text_encoder(tokens).last_hidden_state

            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, t)

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = unet(noisy_latents, t, encoder_hidden_states=enc_hidden).sample
                loss = torch.nn.functional.mse_loss(pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {step} | Loss {loss.item():.5f}")

    unet.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    print("\n🔥 TRAINING FINISHED — LoRA saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
