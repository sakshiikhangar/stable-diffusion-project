
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch import autocast
import numpy as np
from PIL import Image
import os
import time
import gc
from typing import Optional, Tuple, List
from datetime import datetime

from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler
)
import gradio as gr

# CELL 3: Core Generator Class - Part 1 (Initialization)
class StableDiffusionGenerator:    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "auto"):
        try:
            self.device = self._setup_device(device)
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            
            print(f"Initializing Stable Diffusion on {self.device}")
            print(f"Using precision: {self.dtype}")
            
            self.pipe = self._load_pipeline(model_id)
            self.current_scheduler = "euler_a"
            self.schedulers = {
                "euler_a": ("Euler Ancestral", "Fast, good for creative images"),
                "euler": ("Euler", "Deterministic, consistent results"),
                "ddim": ("DDIM", "Classic, good quality, slower"),
                "dpm_solver": ("DPM Solver", "High quality, efficient"),
                "lms": ("LMS", "Linear multistep, stable")
            }
            print("Stable Diffusion Generator Ready!")
        except Exception as e:
            print(f"Initialization Error: {str(e)}")
            raise

# CELL 4: Core Generator Class - Part 2 (Setup Methods)
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"VRAM: {vram_gb:.1f}GB")
            else:
                device = "cpu"
                print("Using CPU (GPU not available)")
        return torch.device(device)
    
    def _load_pipeline(self, model_id: str) -> StableDiffusionPipeline:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            print("Applying Memory Optimizations...")
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("XFormers Attention: Enabled")
            except Exception as e:
                print(f"XFormers: Not available ({e})")
            
            if self.device.type == "cuda":
                try:
                    pipe = pipe.to(self.device)
                    print("Full GPU Loading: Success")
                except RuntimeError as e:
                    print("GPU Memory Limited: Using CPU Offload")
                    pipe.enable_model_cpu_offload()
            else:
                pipe.enable_sequential_cpu_offload()
                print("CPU Sequential Offload: Enabled")
            return pipe
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

# CELL 5: Core Generator Class - Part 3 (Scheduler & Generation)
    def set_scheduler(self, scheduler_name: str) -> bool:
        if scheduler_name not in self.schedulers:
            print(f"Unknown scheduler: {scheduler_name}")
            return False
        if scheduler_name == self.current_scheduler:
            return True
            
        scheduler_map = {
            "euler_a": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "ddim": DDIMScheduler,
            "dpm_solver": DPMSolverMultistepScheduler,
            "lms": LMSDiscreteScheduler
        }
        try:
            scheduler_class = scheduler_map[scheduler_name]
            self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
            self.current_scheduler = scheduler_name
            name, desc = self.schedulers[scheduler_name]
            print(f"Scheduler Changed: {name} ({desc})")
            return True
        except Exception as e:
            print(f"Scheduler Error: {e}")
            return False

# CELL 6: Core Generator Class - Part 4 (Image Generation)
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        scheduler: str = "euler_a"
    ) -> Tuple[Image.Image, dict]:        
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        self.set_scheduler(scheduler)
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
            
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        print(f"Generating: '{prompt[:50]}...'")
        print(f"Size: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}")
        print(f"Seed: {seed}, Scheduler: {scheduler}")
        
        start_time = time.time()
        try:
            with torch.inference_mode():
                if self.device.type == "cuda" and self.dtype == torch.float16:
                    with autocast(self.device.type):
                        result = self.pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator
                        )
                else:
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt if negative_prompt else None,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    )
            
            generation_time = time.time() - start_time
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": scheduler,
                "seed": seed,
                "generation_time": round(generation_time, 2),
                "device": str(self.device),
                "dtype": str(self.dtype)
            }
            print(f"Generated in {generation_time:.2f}s")
            return result.images[0], metadata
            
        except torch.cuda.OutOfMemoryError:
            self._cleanup_memory()
            raise RuntimeError(
                "GPU Out of Memory! Try: reducing image size, fewer steps, "
                "or use CPU mode. Current settings may be too demanding."
            )
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")
        finally:
            self._cleanup_memory()

# CELL 7: Core Generator Class - Part 5 (Utility Methods)
    def _cleanup_memory(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> dict:
        memory_info = {}
        if self.device.type == "cuda":
            memory_info = {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        else:
            memory_info = {"device": "cpu", "note": "CPU memory tracking not available"}
        return memory_info
    
    def save_image(self, image: Image.Image, metadata: dict, output_dir: str = "outputs") -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sd_gen_{timestamp}_s{metadata['seed']}_{metadata['width']}x{metadata['height']}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        
        metadata_file = filepath.replace('.png', '_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write("Stable Diffusion Generation Metadata\n")
            f.write("=" * 40 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"Saved: {filepath}")
        return filepath

# CELL 8: UI Class - Part 1 (Initialization & Generator Setup)
class StableDiffusionUI:
    def __init__(self):
        self.generator = None
        self.gallery_images = []
        self.generation_history = []
    
    def initialize_generator(self, model_choice: str, device_choice: str) -> str:
        try:
            model_map = {
                "Stable Diffusion 1.5 (Recommended)": "runwayml/stable-diffusion-v1-5",
                "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1",
                "Realistic Vision (RealVisXL)": "SG161222/RealVisXL_V4.0"
            }
            device_map = {
                "Auto (Recommended)": "auto",
                "GPU (CUDA)": "cuda", 
                "CPU (Slower)": "cpu"
            }
            model_id = model_map.get(model_choice, "runwayml/stable-diffusion-v1-5")
            device = device_map.get(device_choice, "auto")
            
            self.generator = StableDiffusionGenerator(model_id=model_id, device=device)
            memory_info = self.generator.get_memory_usage()
            memory_text = f"Memory Usage: {memory_info}" if memory_info else "Ready!"
            return f"Model loaded successfully!\n{memory_text}"
        except Exception as e:
            return f"Initialization failed: {str(e)}"

# CELL 9: UI Class - Part 2 (Image Generation Handler)
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        scheduler: str,
        seed: int,
        save_image: bool
    ) -> Tuple[Optional[Image.Image], str, str]:
        if self.generator is None:
            return None, "Please initialize the model first!", ""
        if not prompt.strip():
            return None, "Please enter a prompt!", ""
            
        try:
            seed = None if seed == -1 else int(seed)
            image, metadata = self.generator.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                scheduler=scheduler,
                seed=seed
            )
            
            info_text = self._format_generation_info(metadata)
            saved_path = ""
            if save_image:
                saved_path = self.generator.save_image(image, metadata)
            
            self.generation_history.append(metadata)
            self.gallery_images.append(image)
            
            if len(self.gallery_images) > 10:
                self.gallery_images = self.gallery_images[-10:]
                self.generation_history = self.generation_history[-10:]
            
            return image, info_text, saved_path
        except Exception as e:
            return None, f"Generation failed: {str(e)}", ""

# CELL 10: UI Class - Part 3 (Helper Methods)
    def _format_generation_info(self, metadata: dict) -> str:
        return f"""
Generation Complete!

Parameters Used:
- Prompt: {metadata['prompt'][:100]}{'...' if len(metadata['prompt']) > 100 else ''}
- Size: {metadata['width']} x {metadata['height']} pixels
- Steps: {metadata['steps']} (more steps = higher quality, slower)
- Guidance Scale: {metadata['guidance_scale']} (higher = follows prompt more closely)
- Scheduler: {metadata['scheduler']} 
- Seed: {metadata['seed']} (for reproducible results)

Performance:
- Generation Time: {metadata['generation_time']}s
- Device: {metadata['device']}
- Precision: {metadata['dtype']}
"""
    
    def get_example_prompts(self) -> list:
        return [
            ["a serene mountain landscape at sunrise, photorealistic, highly detailed", "blurry, low quality"],
            ["portrait of a wise old wizard, fantasy art, digital painting", "ugly, deformed"],
            ["cyberpunk cityscape at night, neon lights, futuristic", "daytime, bright"],
            ["cute cartoon cat wearing a hat, kawaii style", "realistic, scary"],
            ["abstract geometric patterns, colorful, modern art", "representational, dull colors"]
        ]
    
    def show_scheduler_info(self, scheduler: str) -> str:
        scheduler_info = {
            "euler_a": "Euler Ancestral: Fast and creative, adds slight randomness for variety",
            "euler": "Euler: Deterministic and consistent, same seed = same result", 
            "ddim": "DDIM: Classic scheduler, high quality but slower",
            "dpm_solver": "DPM Solver: Efficient high-quality generation",
            "lms": "LMS: Linear multistep, very stable results"
        }
        return scheduler_info.get(scheduler, "Scheduler information not available")
    
    def get_memory_info(self) -> str:
        if self.generator is None:
            return "Model not loaded"
        try:
            memory_info = self.generator.get_memory_usage()
            if 'allocated_gb' in memory_info:
                return f"""
GPU Memory Usage:
- Allocated: {memory_info['allocated_gb']:.2f}GB
- Reserved: {memory_info['reserved_gb']:.2f}GB  
- Total Available: {memory_info['total_gb']:.2f}GB
- Usage: {(memory_info['allocated_gb']/memory_info['total_gb']*100):.1f}%
                """
            else:
                return "CPU mode - memory tracking not available"
        except:
            return "Memory info unavailable"

# CELL 11: UI Class - Part 4 (Interface Creation)
    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            title="Educational Stable Diffusion Generator",
            theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown("""
            # Educational Stable Diffusion Text-to-Image Generator
            **Learn Generative AI concepts while creating images!**
            """)
            
            with gr.Tab("Setup & Generation"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Setup")
                        model_choice = gr.Dropdown(
                            choices=[
                                "Stable Diffusion 1.5 (Recommended)",
                                "Stable Diffusion 2.1", 
                                "Realistic Vision (RealVisXL)"
                            ],
                            value="Stable Diffusion 1.5 (Recommended)",
                            label="Model Selection"
                        )
                        device_choice = gr.Dropdown(
                            choices=[
                                "Auto (Recommended)",
                                "GPU (CUDA)",
                                "CPU (Slower)"
                            ],
                            value="Auto (Recommended)", 
                            label="Device Selection"
                        )
                        init_btn = gr.Button("Initialize Model", variant="primary")
                        init_status = gr.Textbox(
                            label="Initialization Status",
                            placeholder="Click Initialize Model to start",
                            lines=3
                        )
                    with gr.Column():
                        gr.Markdown("### System Info")
                        memory_btn = gr.Button("Check Memory Usage")
                        memory_info = gr.Textbox(
                            label="Memory Information",
                            placeholder="Click to check memory usage",
                            lines=6
                        )
                
                gr.Markdown("### Image Generation")
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(
                            label="Prompt (Describe what you want)",
                            placeholder="a beautiful landscape painting, oil on canvas, detailed",
                            lines=3
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt (What to avoid)",
                            placeholder="blurry, low quality, bad anatomy",
                            lines=2
                        )
                        generate_btn = gr.Button("Generate Image", variant="primary", size="lg")
                    with gr.Column():
                        with gr.Accordion("Advanced Settings", open=True):
                            with gr.Row():
                                width = gr.Slider(256, 1024, 512, step=64, label="Width")
                                height = gr.Slider(256, 1024, 512, step=64, label="Height")
                            with gr.Row():
                                steps = gr.Slider(10, 100, 20, step=1, label="Inference Steps")
                                guidance = gr.Slider(1.0, 20.0, 7.5, step=0.5, label="Guidance Scale")
                            scheduler = gr.Dropdown(
                                choices=["euler_a", "euler", "ddim", "dpm_solver", "lms"],
                                value="euler_a",
                                label="Scheduler"
                            )
                            scheduler_info = gr.Textbox(
                                label="Scheduler Information",
                                interactive=False,
                                lines=2
                            )
                            with gr.Row():
                                seed = gr.Number(-1, label="Seed")
                                save_image = gr.Checkbox(True, label="Save Generated Images")
                
                with gr.Row():
                    output_image = gr.Image(label="Generated Image", type="pil")
                with gr.Row():
                    generation_info = gr.Textbox(
                        label="Generation Information",
                        lines=10,
                        interactive=False
                    )
                    saved_path = gr.Textbox(
                        label="Saved File Path",
                        interactive=False
                    )
            
            with gr.Tab("Learning Resources"):
                gr.Markdown("""
                ## Understanding Stable Diffusion
                ### What is Diffusion?
                Diffusion models learn to gradually remove noise from random data.
                ### Key Components:
                **CLIP (Text Encoder)**
                **U-Net (Denoising Network)** 
                **VAE (Variational Autoencoder)**
                **Schedulers**
                ### Parameter Guide:
                **Steps (10-100)**: More steps = higher quality but slower generation
                **Guidance Scale (1-20)**: Higher values make the AI follow your prompt more strictly
                **Seed**: Controls randomness - same seed + settings = same image
                **Resolution**: Higher resolution = more detail but needs more GPU memory
                """)
            
            with gr.Tab("Examples & Gallery"):
                gr.Markdown("### Example Prompts to Try")
                examples = gr.Examples(
                    examples=self.get_example_prompts(),
                    inputs=[prompt, negative_prompt],
                    label="Click any example to load it"
                )
                gr.Markdown("### Recent Generations")
                gallery = gr.Gallery(
                    value=[],
                    label="Your Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto"
                )
            
            # Event handlers
            init_btn.click(
                fn=self.initialize_generator,
                inputs=[model_choice, device_choice],
                outputs=init_status
            )
            generate_btn.click(
                fn=self.generate_image,
                inputs=[prompt, negative_prompt, width, height, steps, guidance, scheduler, seed, save_image],
                outputs=[output_image, generation_info, saved_path]
            ).then(
                fn=lambda: self.gallery_images,
                outputs=gallery
            )
            scheduler.change(
                fn=self.show_scheduler_info,
                inputs=scheduler,
                outputs=scheduler_info
            )
            memory_btn.click(
                fn=self.get_memory_info,
                outputs=memory_info
            )
        
        return interface

# CELL 12: Launch the Application
ui = StableDiffusionUI()
interface = ui.create_interface()
interface.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=7860,
    debug=True,
    show_error=True
)
