import streamlit as st
import replicate

# Title and Instructions
st.title("Drag and Drop API Interaction with Replicate")
st.write("Enter your Replicate API key and select models to trigger the Replicate model calls.")

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

# Check if API key is provided
if not api_key:
    st.warning("Please enter your Replicate API key to proceed.")
else:
    # Create columns for the nodes
    col1, col2, col3 = st.columns(3)

    # First API Node (Model 1)
    with col1:
        st.write("### Node 1 - Replicate Model 1")
        if st.button("Call Model 1"):
            st.write("Model 1 called!")
            output1 = call_replicate_model(api_key, "your-model-name-1", {"input": "sample input"})
            st.write(f"Model 1 Output: {output1}")

    # Second API Node (Model 2)
    with col2:
        st.write("### Node 2 - Replicate Model 2")
        if st.button("Call Model 2"):
            st.write("Model 2 called!")
            output2 = call_replicate_model(api_key, "your-model-name-2", {"input": "another sample input"})
            st.write(f"Model 2 Output: {output2}")

    # Third API Node (Model 3)
    with col3:
        st.write("### Node 3 - Replicate Model 3")
        if st.button("Call Model 3"):
            st.write("Model 3 called!")
            output3 = call_replicate_model(api_key, "your-model-name-3", {"input": "yet another input"})
            st.write(f"Model 3 Output: {output3}")
