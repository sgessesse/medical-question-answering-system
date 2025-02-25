import streamlit as st
import requests

# API endpoint configuration
FASTAPI_URL = "http://localhost:8000/process_query"

# Set up the main UI elements
st.title("Medical Query Assistant")

# Create a form with input and submit button
with st.form(key='query_form'):
    # Text input for the medical question
    query = st.text_input("Ask a medical question:")
    
    # Submit button
    submit_button = st.form_submit_button(label='Submit Query')

# Process the query when form is submitted
if submit_button and query:  # Check both form submission and query existence
    # Send POST request to FastAPI backend
    response = requests.post(FASTAPI_URL, json={"text": query})
    
    if response.status_code == 200:
        data = response.json()
        
        # Display the AI-generated answer
        st.subheader("Answer")
        st.write(data["answer"])
        
        # Display any medical entities detected in the query
        st.subheader("Detected Medical Entities")
        st.table(data["entities"])
        
        # Display the source documents used for the answer
        st.subheader("Source Documents")
        for i, source in enumerate(data["sources"]):
            st.markdown(f"**Document {i+1}**")
            st.write(source["content"])
    else:
        st.error("Error processing query")