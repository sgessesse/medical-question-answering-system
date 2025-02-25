from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_retriever():
    """
    Create a document retriever:
    1. Initialize embeddings model
    2. Load the FAISS index
    3. Configure retrieval parameters
    """
    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load the FAISS index and create a retriever
    return FAISS.load_local(
        "faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True  # Required for local index loading
    ).as_retriever(search_kwargs={"k": 3})  # Return top 3 most relevant documents