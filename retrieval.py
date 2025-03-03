from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def get_retriever():
    """
    Create an enhanced document retriever:
    1. Initialize embeddings model with the same model used to create the index
    2. Load the FAISS index
    3. Configure retrieval with MMR for better relevance and diversity
    """
    # Initialize the embeddings model - IMPORTANT: Must match the model used to create the index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # Use original model to match the index
    )
    
    # Load the FAISS index
    vectorstore = FAISS.load_local(
        "faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True  # Required for local index loading
    )
    
    # Configure the retriever with MMR (Maximum Marginal Relevance)
    # This balances relevance with diversity for better results
    return vectorstore.as_retriever(
        search_type="mmr",  # Use MMR instead of similarity for better diversity
        search_kwargs={
            "k": 5,  # Retrieve more documents initially
            "fetch_k": 10,  # Consider top 10 documents for diversity
            "lambda_mult": 0.7,  # Balance between relevance (1.0) and diversity (0.0)
            "filter": None  # Can be used to filter by metadata if added in future
        }
    )

def get_source_info(document):
    """
    Extract source information from a document:
    - Source name from filename
    - Chunk identifier from metadata
    """
    metadata = document.metadata
    
    # Extract source name (book title) from the file path
    source_name = "Unknown Source"
    if "source" in metadata and metadata["source"]:
        file_path = metadata["source"]
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            # Remove file extension and replace underscores with spaces
            source_name = os.path.splitext(filename)[0].replace('_', ' ')
    
    # Get chunk info for citation purposes
    chunk_id = metadata.get("chunk_id", "unknown section")
    
    return {
        "source_name": source_name,
        "chunk_id": chunk_id
    }