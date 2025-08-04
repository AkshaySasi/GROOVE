# ==============================================================================
# GROOVE: A Generative Refinement Framework Using Vision-Language Feedback Loops
#
# This script implements the core logic for the GROOVE agentic system. It uses
# a multi-agent workflow to autonomously refine text-to-image prompts to match
# a target image.
#
# AGENTS:
# 1. Descriptor (BLIP): Provides an initial text description of a target image.
# 2. Refiner (Gemini): Intelligently refines the prompts based on performance.
# 3. Generator (SDXL): Creates an image from the prompts.
# 4. Comparator (CLIP): Scores the similarity between the generated and target images.
#
# ==============================================================================

# --- Imports ---
import os
import json
import warnings
import torch
import google.generativeai as genai
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from diffusers import StableDiffusionXLPipeline

# --- Suppress specific warnings for cleaner output ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration Constants ---
INITIAL_IMAGE_PATH = "/content/my_image.jpg" # Path to the target image.
OUTPUT_DIR = "output"                        # Directory to save generated images.
MAX_ITERATIONS = 5                           # Maximum number of refinement loops.
SIMILARITY_THRESHOLD = 0.95                  # Exit loop if score exceeds this value.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def configure_llm():
    """
    Configures and initializes the Google Gemini LLM for prompt refinement.
    
    This function retrieves the API key from Colab secrets and initializes the
    generative model.

    Returns:
        genai.GenerativeModel: An initialized Gemini model instance.
        None: If the API key is not found or an error occurs.
    """
    try:
        from google.colab import userdata
        api_key = userdata.get('GEMINI_API_KEY')
        if not api_key:
            raise KeyError("GEMINI_API_KEY not found in Colab secrets.")
        genai.configure(api_key=api_key)
        print("Gemini API configured successfully.")
        return genai.GenerativeModel('gemini-1.5-flash-latest')
    except (ImportError, KeyError) as e:
        print(f"Error: Could not configure Gemini API. {e}")
        print("Please follow the setup instructions in the README to add your API key.")
        return None

def check_setup():
    """
    Verifies that the environment is set up correctly before starting.
    
    Checks for the initial image file and creates the output directory if needed.
    """
    print("--- System Check ---")
    if not os.path.exists(INITIAL_IMAGE_PATH):
        print(f"Error: Initial image not found at '{INITIAL_IMAGE_PATH}'")
        exit()
    print(f"Initial image found: {INITIAL_IMAGE_PATH}")

    if DEVICE == "cpu":
        print("Warning: No GPU detected. The process will be extremely slow.")
    else:
        print(f"Success: GPU detected. Using device: '{DEVICE}'")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(f"Output directory is '{OUTPUT_DIR}'")
    print("--------------------\n")

def load_models():
    """
    Loads all required AI models into memory and onto the specified device.

    Returns:
        dict: A dictionary containing the initialized model instances and processors.
    """
    print("Loading models... This may take several minutes.")
    llm_model = configure_llm()
    if llm_model is None:
        exit() # Exit if API key is not configured

    print("Loading Descriptor: BLIP...")
    descriptor_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    descriptor_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)
    
    print("Loading Generator: Stable Diffusion XL...")
    generator_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    ).to(DEVICE)

    print("Loading Comparator: CLIP...")
    comparator_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    comparator_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print("All models loaded.\n")
    
    return {
        "descriptor": (descriptor_model, descriptor_processor),
        "generator": generator_pipe,
        "comparator": (comparator_model, comparator_processor),
        "llm_refiner": llm_model
    }

def describe_image(image: Image.Image, model, processor):
    """
    Generates a base text description (caption) for a given image using BLIP.

    Args:
        image (PIL.Image.Image): The input image.
        model: The loaded BLIP model.
        processor: The loaded BLIP processor.

    Returns:
        str: The generated text caption for the image.
    """
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_length=75)
    return processor.decode(output[0], skip_special_tokens=True).strip()

def refine_prompts_with_llm(llm_model, base_description, previous_prompt, previous_score):
    """
    Uses a text-based LLM to intelligently refine prompts.

    It generates both a new positive prompt and a negative prompt based on
    the performance of the previous attempt.

    Args:
        llm_model: The initialized Gemini model instance.
        base_description (str): The original, clean description of the target image.
        previous_prompt (str): The positive prompt used in the last iteration.
        previous_score (float): The similarity score from the last iteration.

    Returns:
        dict: A dictionary containing "positive_prompt" and "negative_prompt".
    """
    if llm_model is None: 
        return {"positive_prompt": previous_prompt, "negative_prompt": ""}

    metaprompt = f"""
    You are an expert prompt engineer for text-to-image models. Your task is to refine a positive prompt and generate a helpful negative prompt to make the generated image more visually similar to a ground truth description.

    **Ground Truth Description:**
    "{base_description}"

    **Previous Attempt:**
    - **Prompt Used:** "{previous_prompt}"
    - **Similarity Score (0.0 to 1.0):** {previous_score:.4f}

    **Your Instructions:**
    1. Analyze the "Previous Prompt" and rewrite a new, improved positive prompt. Incorporate specific visual details from the "Ground Truth Description" and add quality keywords (e.g., "photorealistic, 4k, sharp focus").
    2. Based on the "Ground Truth Description", create a helpful negative prompt. List keywords for things to avoid (e.g., if the description is a photo, the negative prompt could be "cartoon, anime, painting, watermark, text, blurry").
    3. Your output **MUST** be a single, valid JSON object with two keys: "positive_prompt" and "negative_prompt". Do not include any other text or markdown formatting.

    **JSON Output:**
    """
    
    try:
        response = llm_model.generate_content(metaprompt)
        # LLMs may wrap their output in markdown fences; this removes them.
        json_str = response.text.strip().replace("```json", "").replace("```", "")
        prompts = json.loads(json_str)
        return prompts
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error during LLM refinement: {e}")
        # Fallback to previous prompts if refinement fails
        return {"positive_prompt": previous_prompt, "negative_prompt": ""}

def generate_image(prompt: str, negative_prompt: str, pipe, generator_seed=42):
    """
    Generates an image using the Stable Diffusion XL pipeline.

    Args:
        prompt (str): The positive prompt describing the desired image.
        negative_prompt (str): The negative prompt describing what to avoid.
        pipe: The initialized Stable Diffusion XL pipeline.
        generator_seed (int): A seed for reproducibility.

    Returns:
        PIL.Image.Image: The generated image.
    """
    generator = torch.Generator(device=DEVICE).manual_seed(generator_seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        num_inference_steps=25
    ).images[0]
    return image

def compare_images(image1: Image.Image, image2: Image.Image, model, processor):
    """
    Compares two images using CLIP and returns a semantic similarity score.

    Args:
        image1 (PIL.Image.Image): The first image (typically the original).
        image2 (PIL.Image.Image): The second image (typically the generated one).
        model: The loaded CLIP model.
        processor: The loaded CLIP processor.

    Returns:
        float: A similarity score between 0.0 and 1.0.
    """
    inputs = processor(text=None, images=[image1, image2], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # Normalize features to compute cosine similarity
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    similarity = (image_features[0] @ image_features[1]).item()
    
    # Scale score from [-1, 1] range to [0, 1] for easier interpretation
    return (similarity + 1) / 2

def main():
    """
    Main execution function to run the GROOVE agentic loop.
    """
    # --- 1. Setup and Model Loading ---
    check_setup()
    models = load_models()
    
    descriptor_model, descriptor_processor = models["descriptor"]
    generator_pipe = models["generator"]
    comparator_model, comparator_processor = models["comparator"]
    llm_refiner = models["llm_refiner"]

    # --- 2. Initialization ---
    initial_image = Image.open(INITIAL_IMAGE_PATH).convert("RGB")
    
    print("Generating base description for reference...")
    base_image_description = describe_image(initial_image, descriptor_model, descriptor_processor)
    print(f"Base Description: \"{base_image_description}\"\n")

    best_positive_prompt = base_image_description
    best_negative_prompt = "blurry, low quality, text, watermark, deformed, ugly, bad anatomy"
    best_image = None
    best_score = -1.0
    
    current_positive_prompt = best_positive_prompt
    current_negative_prompt = best_negative_prompt
    
    # --- 3. Agentic Iteration Loop ---
    print("--- Starting Agentic Iteration Loop ---")
    for i in range(MAX_ITERATIONS):
        print(f"\n===== Iteration {i + 1}/{MAX_ITERATIONS} =====")
        
        # Refine prompts on every iteration after the first one
        if i > 0:
            print("Refining prompts with LLM...")
            refined_prompts = refine_prompts_with_llm(
                llm_refiner,
                base_description=base_image_description,
                previous_prompt=best_positive_prompt,
                previous_score=best_score
            )
            current_positive_prompt = refined_prompts.get("positive_prompt", best_positive_prompt)
            current_negative_prompt = refined_prompts.get("negative_prompt", best_negative_prompt)
        
        # Log current prompts and generate the new image
        print(f"Positive Prompt: \"{current_positive_prompt}\"")
        print(f"Negative Prompt: \"{current_negative_prompt}\"")
        generated_image = generate_image(
            prompt=current_positive_prompt,
            negative_prompt=current_negative_prompt,
            pipe=generator_pipe
        )
        
        # Compare the new image and evaluate the performance
        print("Comparing images...")
        similarity_score = compare_images(initial_image, generated_image, comparator_model, comparator_processor)
        print(f"Similarity Score: {similarity_score:.4f}")

        # If the score has improved, save the new best results
        if similarity_score > best_score:
            print(f"New best score found! Previous best: {best_score:.4f}")
            best_score = similarity_score
            best_positive_prompt = current_positive_prompt
            best_negative_prompt = current_negative_prompt
            best_image = generated_image
            
            save_path = os.path.join(OUTPUT_DIR, f"best_image_iteration_{i+1}.png")
            best_image.save(save_path)
            print(f"Saved new best image to '{save_path}'")

        # Check for early exit if a high enough score is achieved
        if best_score >= SIMILARITY_THRESHOLD:
            print(f"\nSimilarity threshold of {SIMILARITY_THRESHOLD} reached. Halting.")
            break

    # --- 4. Final Output ---
    print("\n--- Agentic Loop Finished ---")
    print(f"Final Best Score: {best_score:.4f}")
    print(f"Final Best Positive Prompt: \"{best_positive_prompt}\"")
    print(f"Final Best Negative Prompt: \"{best_negative_prompt}\"")
    
    final_image_path = os.path.join(OUTPUT_DIR, "final_best_image.png")
    if best_image:
        best_image.save(final_image_path)
        print(f"The best overall image has been saved to '{final_image_path}'")

# --- Script Execution ---
if __name__ == "__main__":
    # This block ensures the main function is called only when the script is executed directly
    main()