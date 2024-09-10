import streamlit as st
from streamlit_elements import elements, dashboard, mui

# Main interactive node-based UI using Streamlit Elements
with elements("main"):
    # Create a grid dashboard layout (draggable, resizable)
    grid = dashboard.Grid(cols=12, rowHeight=160, compactType="vertical", draggableHandle=".drag-handle")

    # Example Node 1: User input and button
    with grid.item("node1", 0, 0, 6, 2):
        with mui.Paper(elevation=3):
            st.write("### Node 1 - Input and Button")
            user_input = st.text_input("Enter some text:", key="user_input")
            if st.button("Submit"):
                st.write(f"You entered: {user_input}")

    # Example Node 2: Counter with buttons
    with grid.item("node2", 6, 0, 6, 2):
        with mui.Paper(elevation=3):
            st.write("### Node 2 - Counter")
            if 'counter' not in st.session_state:
                st.session_state.counter = 0
            st.write(f"Counter value: {st.session_state.counter}")
            if st.button("Increment"):
                st.session_state.counter += 1
            if st.button("Decrement"):
                st.session_state.counter -= 1

    # Example Node 3: Dropdown and selection display
    with grid.item("node3", 0, 2, 6, 2):
        with mui.Paper(elevation=3):
            st.write("### Node 3 - Dropdown Example")
            options = ["Option 1", "Option 2", "Option 3"]
            selected = st.selectbox("Choose an option:", options)
            st.write(f"You selected: {selected}")

    # Example Node 4: Image upload and display
    with grid.item("node4", 6, 2, 6, 2):
        with mui.Paper(elevation=3):
            st.write("### Node 4 - Image Upload")
            uploaded_file = st.file_uploader("Upload an image")
            if uploaded_file is not None:
                st.image(uploaded_file)

    # Save and display the grid layout
    grid.save()
