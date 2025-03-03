import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io

# API endpoint configuration
FASTAPI_URL = "http://localhost:8000/process_query"

# Page configuration
st.set_page_config(
    page_title="Medical Q&A Assistant",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border: 1px solid #eaecef;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .entity-container {
        display: inline-block;
        background-color: #e9f7ef;
        border-radius: 15px;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .answer-container {
        margin-top: 1.5rem;
    }
    /* Hide sidebar expander in production */
    .css-eh5xgm.e1fqkh3o3 {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Set up the main UI elements
st.title("🩺 Medical Query Assistant")
st.markdown("Get evidence-based answers to your medical questions using advanced medical knowledge retrieval")

# Create a layout with main content and sidebar
main_col, sidebar_col = st.columns([3, 1])

# Put the sidebar content in the second column
with sidebar_col:
    st.subheader("About")
    st.info("""
    This application uses a retrieval-augmented generation approach to answer medical questions:
    
    1. Your question is analyzed for medical entities
    2. Relevant information is retrieved from medical textbooks
    3. An AI model generates an evidence-based answer
    
    All answers are sourced from established medical literature.
    """)
    
    st.subheader("Sources")
    st.markdown("""
    The system draws information from multiple medical textbooks including:
    
    - Harrison's Internal Medicine
    - Gray's Anatomy
    - DSM-5 Psychiatry
    - Robbins Pathology
    - And many more...
    """)

# Main content area
with main_col:
    # Create a form with input and submit button
    with st.form(key='query_form'):
        # Text input for the medical question
        query = st.text_area("Enter your medical question:", 
                            height=100, 
                            placeholder="Example: What are the risk factors for developing type 2 diabetes?")
        
        # Submit button
        submit_button = st.form_submit_button(label='Submit Query')

    # Process the query when form is submitted
    if submit_button and query:  # Check both form submission and query existence
        with st.spinner("Processing your question..."):
            # Send POST request to FastAPI backend
            response = requests.post(FASTAPI_URL, json={"text": query})
            
            if response.status_code == 200:
                data = response.json()
                
                # Display the AI-generated answer directly below the form
                st.subheader("Answer")
                st.markdown(data["answer"])
                
                # Display source information
                st.subheader("Sources")
                
                for i, source in enumerate(data["sources"]):
                    with st.expander(f"Source {i+1}: {source['source_name']}"):
                        # Display location information if available
                        if source["location"]:
                            st.markdown(f"**Location:** {source['location']}")
                        
                        # Display preview of the source content
                        st.markdown("**Preview:**")
                        st.markdown(source["content"])
                
                # Display any medical entities detected in the query
                if data["entities"]:
                    st.subheader("Detected Medical Entities")
                    
                    # Prepare data for table display
                    entity_df = pd.DataFrame(data["entities"])
                    entity_df.columns = ["Term", "Type"]
                    
                    # Display entities as a table
                    st.table(entity_df)
            else:
                st.error("Error processing query. Please try again.")

# Footer
st.markdown("---")
st.markdown("*This system provides information for educational purposes only. Always consult a healthcare professional for medical advice.*")