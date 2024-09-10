import streamlit as st
import replicate

# Input for the API Key
api_key = st.text_input("Enter your Replicate API Key", type="password")

# Initialize Replicate client
if api_key:
    client = replicate.Client(api_token=api_key)

    # Model Selection
    model_type = st.selectbox(
        "Choose a Model",
        ["LLaMA 70b (Text Generation)", "Flux Pro (Art Generation)", "Image Upscaler"]
    )

    # Text generation with LLaMA 70b
    if model_type == "LLaMA 70b (Text Generation)":
        prompt_1 = st.text_area("Enter the text prompt for LLaMA 70b:")
        max_length = st.slider("Max Length of Generated Text", 50, 500, 150)
        if st.button("Generate Text"):
            if prompt_1:
                model = client.models.get("replicate/llama-70b")
                output_1 = model.predict(prompt=prompt_1, max_length=max_length)
                st.write(f"Generated Text: {output_1}")
            else:
                st.warning("Please enter a text prompt.")

    # Flux Pro for Art Generation
    if model_type == "Flux Pro (Art Generation)":
        prompt_2 = st.text_area("Enter the art prompt for Flux Pro:")
        style = st.selectbox("Choose an art style", ["abstract", "cyberpunk", "fantasy", "realistic"])
        guidance_scale = st.slider("Guidance Scale", 0.5, 20.0, 7.5)
        if st.button("Generate Art"):
            if prompt_2:
                model = client.models.get("black-forest-labs/flux-pro")
                output_2 = model.predict(prompt=prompt_2, style=style, guidance_scale=guidance_scale)
                st.image(output_2, caption="Generated Art")
            else:
                st.warning("Please enter an art prompt.")

    # Image Upscaler
    if model_type == "Image Upscaler":
        uploaded_file = st.file_uploader("Upload an image to upscale", type=["png", "jpg", "jpeg"])
        upscale_factor = st.slider("Upscale Factor", 2, 8, 4)
        if st.button("Upscale Image"):
            if uploaded_file:
                model = client.models.get("stability-ai/stable-diffusion-x4-upscaler")
                output_3 = model.predict(image=uploaded_file, upscale_factor=upscale_factor)
                st.image(output_3, caption="Upscaled Image")
            else:
                st.warning("Please upload an image.")
