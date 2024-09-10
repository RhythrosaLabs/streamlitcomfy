import streamlit as st
from streamlit_elements import elements, mui, dashboard

# Define the layout using dashboard.Layout
with elements("node_ui"):
    layout = dashboard.Layout()

    # Add draggable cards (nodes)
    with layout.item("node1", x=0, y=0, w=3, h=2):
        mui.Card(
            children=[
                mui.CardHeader(title="Node 1", className="drag-handle"),
                mui.CardContent("Content of Node 1"),
                mui.Button("Call Node 1")
            ]
        )

    with layout.item("node2", x=3, y=0, w=3, h=2):
        mui.Card(
            children=[
                mui.CardHeader(title="Node 2", className="drag-handle"),
                mui.CardContent("Content of Node 2"),
                mui.Button("Call Node 2")
            ]
        )
