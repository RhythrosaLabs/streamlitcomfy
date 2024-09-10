import streamlit as st
import replicate

# Title and Instructions
st.title("Chained Replicate Model API Calls")
st.write("Select models and provide input text for each node. Outputs from one node can be passed to the next.")

# Input field for Replicate API key
api_key = st.text_input("Enter your Replicate API key", type="password")

# Function to call a Replicate model
def call_replicate_model(api_key, model_name, inputs):
    try:
        client = replicate.Client(api_token=api_key)
        model = client.models.get(model_name)
        return model.predict(**inputs)
    except Exception as e:
        return f"Error: {str(e)}"

# List of example model names from Replicate (replace with actual model names)
available_models = ["model-name-1", "model-name-2", "model-name-3"]

# Check if API key is provided
if not api_key:
    st.warning("Please enter your Replicate API key to proceed.")
else:
    # Create containers for each model node
    output_1 = None
    output_2 = None
    output_3 = None

    # First Node
    with st.container():
        st.write("### Node 1")
        model_1 = st.selectbox("Select Model for Node 1", available_models, key="model_1")
        input_1 = st.text_input("Input for Model 1", value="Sample input for Model 1", key="input_1")

        if st.button("Run Model 1"):
            output_1 = call_replicate_model(api_key, model_1, {"input": input_1})
            st.write(f"Output from Model 1: {output_1}")

    # Second Node (chainable input from Node 1)
    with st.container():
        st.write("### Node 2 (Chained from Node 1)")
        model_2 = st.selectbox("Select Model for Node 2", available_models, key="model_2")
        input_2 = st.text_input("Input for Model 2", value=output_1 if output_1 else "Sample input for Model 2", key="input_2")

        if st.button("Run Model 2"):
            output_2 = call_replicate_model(api_key, model_2, {"input": input_2})
            st.write(f"Output from Model 2: {output_2}")

    # Third Node (chainable input from Node 2)
    with st.container():
        st.write("### Node 3 (Chained from Node 2)")
        model_3 = st.selectbox("Select Model for Node 3", available_models, key="model_3")
        input_3 = st.text_input("Input for Model 3", value=output_2 if output_2 else "Sample input for Model 3", key="input_3")

        if st.button("Run Model 3"):
            output_3 = call_replicate_model(api_key, model_3, {"input": input_3})
            st.write(f"Output from Model 3: {output_3}")
