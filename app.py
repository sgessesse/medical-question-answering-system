from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from retrieval import get_retriever, get_source_info
from generation import get_answer_chain
from query_processing import extract_medical_entities, expand_query
from fastapi.middleware.cors import CORSMiddleware
import traceback
import sys

# Initialize FastAPI application
app = FastAPI(
    title="Medical Question Answering API",
    description="An API for answering medical questions using a retrieval-augmented generation approach",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch and log all errors"""
    error_msg = f"Exception occurred: {str(exc)}"
    traceback_str = traceback.format_exc()
    print(error_msg)
    print(f"Traceback: {traceback_str}")
    return JSONResponse(
        status_code=500,
        content={"error": error_msg, "traceback": traceback_str},
    )

# Initialize the document retriever and answer generation chain
try:
    retriever = get_retriever()
    answer_chain = get_answer_chain()
    print("Successfully initialized retriever and answer chain")
except Exception as e:
    print(f"Error initializing components: {str(e)}")
    print(traceback.format_exc())
    # We'll let the app start anyway and handle errors at request time

# Define the request model for query processing
class QueryRequest(BaseModel):
    text: str

@app.post("/process_query")
async def process_query(request: QueryRequest):
    """
    Process an incoming medical query:
    1. Extract medical entities from the query
    2. Use entities to expand the query for better retrieval
    3. Retrieve relevant documents
    4. Generate an answer based on the documents
    5. Return the answer with enhanced source information
    """
    try:
        # Get the query text from the request
        query = request.text
        print(f"Received query: {query}")
        
        # Extract medical entities from the query
        try:
            entities = extract_medical_entities(query)
            print(f"Extracted entities: {entities}")
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            entities = []
        
        # Use entities to expand the query for better retrieval
        try:
            expanded_query = expand_query(query, entities)
            print(f"Expanded query: {expanded_query}")
        except Exception as e:
            print(f"Error expanding query: {str(e)}")
            expanded_query = query
        
        # Retrieve relevant documents based on the expanded query
        try:
            docs = retriever.invoke(expanded_query)
            print(f"Retrieved {len(docs)} documents")
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            raise
        
        # Generate an answer using the retrieved documents
        try:
            answer = answer_chain({"context": docs, "question": query})
            print("Generated answer successfully")
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            raise
        
        # Format source information for response
        sources = []
        try:
            for doc in docs:
                source_info = get_source_info(doc)
                source_data = {
                    "content": doc.page_content[:600] + "..." if len(doc.page_content) > 600 else doc.page_content,  # Increased preview length for better context
                    "source_name": source_info["source_name"],
                    "location": ""
                }
                
                # Add section/page information if available
                if "section" in doc.metadata:
                    source_data["location"] += f"Section: {doc.metadata['section']} "
                if "page" in doc.metadata:
                    source_data["location"] += f"Page: {doc.metadata['page']}"
                    
                sources.append(source_data)
            print(f"Formatted {len(sources)} sources")
        except Exception as e:
            print(f"Error formatting sources: {str(e)}")
            # Continue with empty sources if there's an error
            sources = []
        
        # Return the processed results with enhanced information
        return {
            "answer": answer,
            "entities": entities,
            "sources": sources,
            "expanded_query": expanded_query if expanded_query != query else None
        }
    except Exception as e:
        print(f"Unhandled error in process_query: {str(e)}")
        print(traceback.format_exc())
        raise

@app.get("/")
async def root():
    return {"message": "Medical Question Answering System API. Use /process_query endpoint to ask questions."}