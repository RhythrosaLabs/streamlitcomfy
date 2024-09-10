import streamlit as st
import streamlit_dnd as dnd
import replicate

# Define the API key for Replicate (replace with your actual API key)
REPLICATE_API_TOKEN = "your_replicate_api_token"

# Function to call a Replicate model
def call_replicate_model(model_name, inputs):
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    model = client.models.get(model_name)
    return model.predict(**inputs)

# Title and Instructions
st.title("Drag and Drop API Interaction")
st.write("Drag the API nodes into the drop zones to trigger the Replicate model calls.")

# First draggable component
st.write("### Node 1 - Replicate Model 1")
result1 = dnd.drop_target("Drag Model 1 here", key="drop_1")
if result1:
    st.write("Model 1 called!")
    # Trigger a Replicate model call here
    output1 = call_replicate_model("your-model-name-1", {"input": "sample input"})
    st.write(f"Model 1 Output: {output1}")

# Second draggable component
st.write("### Node 2 - Replicate Model 2")
result2 = dnd.drop_target("Drag Model 2 here", key="drop_2")
if result2:
    st.write("Model 2 called!")
    # Trigger another Replicate model call here
    output2 = call_replicate_model("your-model-name-2", {"input": "another sample input"})
    st.write(f"Model 2 Output: {output2}")

# Add more draggable components if necessary
st.write("### Node 3 - Replicate Model 3")
result3 = dnd.drop_target("Drag Model 3 here", key="drop_3")
if result3:
    st.write("Model 3 called!")
    # Trigger yet another Replicate model call
    output3 = call_replicate_model("your-model-name-3", {"input": "yet another input"})
    st.write(f"Model 3 Output: {output3}")
