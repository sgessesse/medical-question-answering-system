from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import re

def extract_section_info(text):
    """
    Extract section headings and potential page numbers from text.
    Returns best heading match or None if not found.
    """
    # Look for section headers (uppercase, followed by newline)
    section_pattern = r'([A-Z][A-Z\s]+(?:\n|:))'
    # Look for potential page numbers
    page_pattern = r'PAGE\s+(\d+)'
    
    # Try to find chapter/section headers
    sections = re.findall(section_pattern, text)
    section = sections[-1].strip() if sections else None
    
    # Try to find page numbers
    pages = re.findall(page_pattern, text, re.IGNORECASE)
    page = pages[-1] if pages else None
    
    return {
        "section": section,
        "page": page
    }

def ingest_docs():
    """
    Process medical documents from the sample_docs directory:
    1. Load text files
    2. Split them into chunks with enhanced metadata
    3. Create embeddings
    4. Store in a FAISS vector database
    """
    # Load sample medical documents
    documents = []
    for filename in os.listdir('sample_docs/'):
        if filename.endswith('.txt'):
            try:
                # Get the full path to use in metadata
                file_path = os.path.join('sample_docs/', filename)
                
                # Try loading with UTF-8 encoding first
                loader = TextLoader(
                    file_path,
                    encoding='utf-8',
                    autodetect_encoding=True  # Attempt to detect file encoding
                )
                
                # Load documents with source attribution
                file_docs = loader.load()
                
                # Set source metadata for citation
                for doc in file_docs:
                    doc.metadata["source"] = file_path
                    # Extract book title from filename for better citations
                    doc.metadata["book_title"] = os.path.splitext(filename)[0].replace('_', ' ')
                
                documents.extend(file_docs)
                print(f"Successfully loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                # Fallback to latin-1 encoding if UTF-8 fails
                try:
                    loader = TextLoader(
                        file_path,
                        encoding='latin-1'
                    )
                    file_docs = loader.load()
                    
                    # Set source metadata for citation
                    for doc in file_docs:
                        doc.metadata["source"] = file_path
                        doc.metadata["book_title"] = os.path.splitext(filename)[0].replace('_', ' ')
                    
                    documents.extend(file_docs)
                    print(f"Successfully loaded {filename} with latin-1 encoding")
                except Exception as e:
                    print(f"Failed to load {filename} with both encodings: {str(e)}")

    # Split documents into smaller chunks with better metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Number of characters per chunk
        chunk_overlap=200,  # Overlap between chunks to maintain context
        separators=["\n\n", "\n", ". ", " ", ""]  # Priority of separators for better context
    )
    
    # Process documents with enhanced metadata
    processed_docs = []
    for i, doc in enumerate(documents):
        chunks = text_splitter.split_text(doc.page_content)
        
        for j, chunk in enumerate(chunks):
            # Extract section information from chunk
            section_info = extract_section_info(chunk)
            
            # Create metadata for this chunk
            metadata = doc.metadata.copy()
            metadata["chunk_id"] = f"chunk_{j+1}"
            if section_info["section"]:
                metadata["section"] = section_info["section"]
            if section_info["page"]:
                metadata["page"] = section_info["page"]
            
            # Create a new document with the enhanced metadata
            processed_docs.append(
                text_splitter.create_documents(
                    [chunk], 
                    [metadata]
                )[0]
            )
    
    print(f"Split {len(documents)} documents into {len(processed_docs)} chunks")

    # Create embeddings using a pre-trained model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # Improved embedding model
    )
    
    # Create and save the FAISS vector store
    vector_store = FAISS.from_documents(processed_docs, embeddings)
    vector_store.save_local("faiss_index")
    print("Vector store saved successfully")

if __name__ == "__main__":
    ingest_docs()
