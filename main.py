import streamlit as st
import replicate

# Input for the API Key
api_key = st.text_input("Enter your Replicate API Key", type="password")

# Initialize Replicate client
if api_key:
    client = replicate.Client(api_token=api_key)

    # Select the model
    model_type = st.selectbox(
        "Choose a Model Type",
        ["LLaMA 70b (Text Generation)", "Image Upscaler", "Flux (Art Generation)"]
    )

    # Text generation with LLaMA 70b
    if model_type == "LLaMA 70b (Text Generation)":
        st.write("### LLaMA 70b - Text Generation")
        prompt = st.text_area("Enter the text prompt:")
        max_length = st.slider("Max Length of Generated Text", 50, 500, 150)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.75)

        if st.button("Generate Text"):
            if prompt:
                model = client.models.get("replicate/llama-70b")
                output = model.predict(prompt=prompt, max_length=max_length, temperature=temperature)
                st.write(f"Generated Text: {output}")
            else:
                st.warning("Please enter a prompt.")

    # Image Upscaler
    elif model_type == "Image Upscaler":
        st.write("### Image Upscaler")
        uploaded_file = st.file_uploader("Upload an image to upscale", type=["png", "jpg", "jpeg"])
        upscale_factor = st.slider("Upscale Factor", 2, 8, 4)

        if st.button("Upscale Image"):
            if uploaded_file:
                model = client.models.get("stability-ai/stable-diffusion-x4-upscaler")
                output = model.predict(image=uploaded_file, upscale_factor=upscale_factor)
                st.image(output, caption="Upscaled Image")
            else:
                st.warning("Please upload an image.")

    # Flux (Art Generation)
    elif model_type == "Flux (Art Generation)":
        st.write("### Flux - Art Generation")
        prompt = st.text_area("Enter the art prompt:")
        style = st.selectbox("Choose an art style", ["abstract", "cyberpunk", "fantasy", "realistic"])
        guidance_scale = st.slider("Guidance Scale", 0.5, 20.0, 7.5)

        if st.button("Generate Art"):
            if prompt:
                model = client.models.get("flux/art-generator")
                output = model.predict(prompt=prompt, style=style, guidance_scale=guidance_scale)
                st.image(output, caption="Generated Art")
            else:
                st.warning("Please enter an art prompt.")
