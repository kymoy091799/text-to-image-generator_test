import os
import sys
import base64
import time
import json
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alternative_handler")

# Print startup info
logger.info("=" * 40)
logger.info("ALTERNATIVE HANDLER STARTING")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info("=" * 40)

# Import RunPod with error handling
try:
    import runpod
    logger.info("Successfully imported runpod")
except Exception as e:
    logger.error(f"Error importing runpod: {e}")
    import traceback
    traceback.print_exc()
    raise

# Global variable to hold the model
model = None

def load_model():
    """Load an alternative, publicly available model (Stable Diffusion XL)"""
    global model
    
    # Already loaded check
    if model is not None:
        logger.info("Model already loaded, reusing existing model")
        return model
    
    try:
        # Import dependencies inside function to catch and report errors
        logger.info("Importing torch and diffusers")
        import torch
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        
        # Log environment info
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Device selection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Model loading with clear progress indication
        logger.info("Starting model download and loading")
        logger.info("This may take several minutes on first run")
        
        start_time = time.time()
        
        # Load Stable Diffusion XL - a publicly available model
        logger.info("Loading model: stabilityai/stable-diffusion-xl-base-1.0")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        
        # Move to device
        logger.info(f"Moving model to {device}")
        pipeline = pipeline.to(device)
        
        # Enable memory optimizations for CUDA
        if device == "cuda":
            logger.info("Enabling memory optimizations")
            pipeline.enable_attention_slicing()
        
        # Use a faster scheduler
        logger.info("Setting up DPM++ scheduler")
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        model = pipeline
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        model = None
        raise RuntimeError(f"Failed to load model: {str(e)}")

def to_base64_string(image):
    """Convert a PIL image to a base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def handler(event):
    """Handle the serverless request to generate an image from text."""
    try:
        # Log the event, but omit potentially large data
        logger.info(f"Received event type: {type(event)}")
        if isinstance(event, dict):
            logger.info(f"Event keys: {list(event.keys())}")
            
        # Get input parameters from the request
        input_data = event.get("input", {})
        logger.info(f"Input data: {json.dumps(input_data)}")
        
        # Load model if not already loaded
        logger.info("Checking model status")
        global model
        
        if model is None:
            logger.info("Model not loaded, loading now...")
            model = load_model()
            if model is None:
                error_msg = "Failed to load model after multiple attempts"
                logger.error(error_msg)
                return {"error": error_msg}
        
        # Extract parameters with defaults
        prompt = input_data.get("prompt", "A beautiful landscape")
        negative_prompt = input_data.get("negative_prompt", "")
        height = int(input_data.get("height", 1024))
        width = int(input_data.get("width", 1024))
        num_inference_steps = int(input_data.get("num_inference_steps", 30))
        guidance_scale = float(input_data.get("guidance_scale", 7.5))
        
        # Generate image
        logger.info(f"Generating image with prompt: '{prompt}'")
        start_time = time.time()
        
        # Simple generation with fewer parameters to avoid errors
        result = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Generated image in {generation_time:.2f} seconds")
        
        # Convert image to base64
        logger.info("Converting image to base64")
        image_data = to_base64_string(result.images[0])
        
        # Return the result
        return {
            "image": image_data,
            "metrics": {
                "generation_time": generation_time
            }
        }
    
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return {"error": error_msg}

# Startup message before starting the serverless function
logger.info("Starting runpod serverless with alternative handler")

# Start the serverless function
try:
    runpod.serverless.start({"handler": handler})
except Exception as e:
    logger.error(f"Failed to start serverless function: {e}")
    import traceback
    traceback.print_exc()