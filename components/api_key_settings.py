import streamlit as st

def display_api_key_settings():
    st.header("API Key Management")
    st.caption("Securely store and manage your Replicate API key for model access.")
    api_key = st.text_input("Enter your Replicate API Key", type="password", help="Your API key is required to access Replicate models.")
    
    if api_key:
        st.session_state['replicate_api_key'] = api_key
        st.success("API Key saved successfully!")
