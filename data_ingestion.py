from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def ingest_docs():
    """
    Process medical documents from the sample_docs directory:
    1. Load text files
    2. Split them into chunks
    3. Create embeddings
    4. Store in a FAISS vector database
    """
    # Load sample medical documents
    documents = []
    for filename in os.listdir('sample_docs/'):
        if filename.endswith('.txt'):
            try:
                # Try loading with UTF-8 encoding first
                loader = TextLoader(
                    os.path.join('sample_docs/', filename),
                    encoding='utf-8',
                    autodetect_encoding=True  # Attempt to detect file encoding
                )
                documents.extend(loader.load())
                print(f"Successfully loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                # Fallback to latin-1 encoding if UTF-8 fails
                try:
                    loader = TextLoader(
                        os.path.join('sample_docs/', filename),
                        encoding='latin-1'
                    )
                    documents.extend(loader.load())
                    print(f"Successfully loaded {filename} with latin-1 encoding")
                except Exception as e:
                    print(f"Failed to load {filename} with both encodings: {str(e)}")

    # Split documents into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Number of characters per chunk
        chunk_overlap=200  # Overlap between chunks to maintain context
    )
    texts = text_splitter.split_documents(documents)

    # Create embeddings using a pre-trained model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create and save the FAISS vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local("faiss_index")

if __name__ == "__main__":
    ingest_docs()
