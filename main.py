import streamlit as st
from streamlit_elements import elements, mui, dashboard

# Create a dashboard layout
with elements("node_ui"):
    # Define the layout with the 'dashboard.Grid' method
    layout = dashboard.Grid(
        breakpointCols={"lg": 12, "md": 10, "sm": 6, "xs": 4, "xxs": 2},  # Define column breakpoints
        rowHeight=150,  # Set row height for the grid
        containers=True  # Allow draggable containers (nodes)
    )

    # Define a draggable card (node)
    with layout.item(key="node1", x=0, y=0, w=3, h=2):
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
    
    with layout.item(key="node2", x=3, y=0, w=3, h=2):
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
