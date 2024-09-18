import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load the model on startup (outside the function)
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Check for GPU

# Streamlit App Title
st.title("Text to Image Generator")

@st.cache_resource  # Cache the model to avoid reloading it on every interaction
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)  # Load the model onto the correct device
    return pipe

pipe = load_model()

# Resolution choices for user input
resolution_options = [256, 512, 768]
resolution = st.selectbox("Choose Image Resolution", resolution_options)

# Inference steps for user input
inference_steps_options = [20, 30, 40]
inference_steps = st.selectbox("Choose Inference Steps (Higher = Better Quality)", inference_steps_options)

# Input text prompt from user
prompt = st.text_input("Enter a text prompt to generate an image:")

# Button to generate image
if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            # Generate image on the GPU or CPU depending on availability
            image = pipe(prompt, num_inference_steps=inference_steps, height=resolution, width=resolution).images[0]

            # Display the image
            st.image(image, caption="Generated Image", use_column_width=True)

            # Save the image if needed
            image.save("generated_image.png")
            st.success("Image generated and saved!")
    else:
        st.warning("Please enter a prompt to generate an image.")
