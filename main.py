import streamlit as st
from streamlit_elements import elements, mui, dashboard
import replicate

# Define your Replicate API key
REPLICATE_API_TOKEN = "your_replicate_api_token"

# Function to call a Replicate model
def call_replicate_model(model_name, inputs):
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    model = client.models.get(model_name)
    output = model.predict(**inputs)
    return output

# Create a draggable and resizable dashboard
with elements("node_ui"):
    layout = dashboard.Grid(
        breakpointCols={"lg": 12, "md": 10, "sm": 6, "xs": 4, "xxs": 2},
        rowHeight=150,
    )

    # Define draggable nodes (API Node examples)
    with layout.item(key="node1", x=0, y=0, w=3, h=2):
        mui.Card(
            children=[
                mui.CardHeader(title="Replicate Node 1", className="drag-handle"),
                mui.CardContent("This node calls a Replicate model."),
                mui.Button("Call Model 1", onClick=lambda: st.session_state.update({"output1": call_replicate_model("model1", {"input": "value"})}))
            ]
        )

    with layout.item(key="node2", x=3, y=0, w=3, h=2):
        mui.Card(
            children=[
                mui.CardHeader(title="Replicate Node 2", className="drag-handle"),
                mui.CardContent("This node calls another Replicate model."),
                mui.Button("Call Model 2", onClick=lambda: st.session_state.update({"output2": call_replicate_model("model2", {"input": "value2"})}))
            ]
        )

    # Display outputs from models
    if "output1" in st.session_state:
        st.write("Output from Model 1: ", st.session_state["output1"])
    if "output2" in st.session_state:
        st.write("Output from Model 2: ", st.session_state["output2"])
