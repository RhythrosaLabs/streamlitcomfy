import streamlit as st
from streamlit_elements import elements, mui, dashboard

# Create a dashboard layout
with elements("node_ui"):
    layout = dashboard.GridLayout(
        id="node-layout",
        cols=12,
        rowHeight=150,
        draggableHandle=".drag-handle"
    )

    # Define a draggable card (node)
    with layout.item("node1", 0, 0, 3, 2):
        mui.Card(
            children=[
                mui.CardHeader(
                    title="API Node 1",
                    className="drag-handle"
                ),
                mui.CardContent("Select options for API"),
                mui.Button("Call API")
            ]
        )
    
    with layout.item("node2", 3, 0, 3, 2):
        mui.Card(
            children=[
                mui.CardHeader(
                    title="API Node 2",
                    className="drag-handle"
                ),
                mui.CardContent("Select options for API 2"),
                mui.Button("Call API 2")
            ]
        )
