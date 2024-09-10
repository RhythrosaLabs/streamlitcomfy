import streamlit as st
import replicate

# Input for the API Key
api_key = st.text_input("Enter your Replicate API Key", type="password")

# Initialize Replicate client
if api_key:
    client = replicate.Client(api_token=api_key)

    # Select the first model (for text generation or art prompt)
    model_type_1 = st.selectbox(
        "Choose First Model (Output can be used in the next step)",
        ["LLaMA 70b (Text Generation)", "Flux (Art Generation)"]
    )

    # Output variable for first model's result
    output_1 = None

    # If LLaMA 70b is selected
    if model_type_1 == "LLaMA 70b (Text Generation)":
        st.write("### LLaMA 70b - Text Generation")
        prompt_1 = st.text_area("Enter the text prompt for LLaMA 70b:")
        max_length = st.slider("Max Length of Generated Text", 50, 500, 150)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.75)

        if st.button("Generate Text"):
            if prompt_1:
                model = client.models.get("replicate/llama-70b")
                output_1 = model.predict(prompt=prompt_1, max_length=max_length, temperature=temperature)
                st.write(f"Generated Text: {output_1}")
            else:
                st.warning("Please enter a prompt.")

    # If Flux is selected
    if model_type_1 == "Flux (Art Generation)":
        st.write("### Flux - Art Generation")
        prompt_1 = st.text_area("Enter the art prompt for Flux:")
        style = st.selectbox("Choose an art style", ["abstract", "cyberpunk", "fantasy", "realistic"])
        guidance_scale = st.slider("Guidance Scale", 0.5, 20.0, 7.5)

        if st.button("Generate Art"):
            if prompt_1:
                model = client.models.get("flux/art-generator")
                output_1 = model.predict(prompt=prompt_1, style=style, guidance_scale=guidance_scale)
                st.image(output_1, caption="Generated Art")
            else:
                st.warning("Please enter an art prompt.")

    # Chainable second model
    if output_1:
        st.write("### Chain Second Model (Process Output from First Step)")
        
        # Select the second model
        model_type_2 = st.selectbox(
            "Choose Second Model to Process Output",
            ["Image Upscaler", "Flux (Art Generation from Text)"]
        )
        
        # Upscale if the output is an image
        if model_type_2 == "Image Upscaler" and isinstance(output_1, bytes):
            upscale_factor = st.slider("Upscale Factor", 2, 8, 4)
            if st.button("Upscale Image"):
                model = client.models.get("stability-ai/stable-diffusion-x4-upscaler")
                output_2 = model.predict(image=output_1, upscale_factor=upscale_factor)
                st.image(output_2, caption="Upscaled Image")
        
        # Art Generation using Flux if the first output is text
        elif model_type_2 == "Flux (Art Generation from Text)" and isinstance(output_1, str):
            style_2 = st.selectbox("Choose an art style", ["abstract", "cyberpunk", "fantasy", "realistic"])
            guidance_scale_2 = st.slider("Guidance Scale for Second Model", 0.5, 20.0, 7.5)
            if st.button("Generate Art from Generated Text"):
                model = client.models.get("flux/art-generator")
                output_2 = model.predict(prompt=output_1, style=style_2, guidance_scale=guidance_scale_2)
                st.image(output_2, caption="Generated Art from Text")
